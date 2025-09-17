package main

import (
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/tfriedel6/canvas"
	"github.com/tfriedel6/canvas/sdlcanvas"
)

const (
	screenW  = 1200
	screenH  = 800
	numParts = 800 // bump up density for richer tissue feel

	// Neighborhood radii
	neighborRadiusSame  = 20.0 // slightly smaller → cells mostly interact with close neighbors
	neighborRadiusCross = 22.0 // slightly larger → lets populations overlap/mix at boundaries

	// LIFE-LIKE PRESET (tuned)
	preferredSpacing = 13.5  // tighter packing, feels like crowded cytoplasm
	baseTension      = 0.030 // firmer membranes, keeps cells round
	polygonBias      = 0.025 // stronger hexagonal bias → honeycomb tissue alignment
	pressureScale    = 1.15  // stronger size regulation, prevents collapse
	noiseStrength    = 0.10  // gentle Brownian jiggle, draggy/viscous
	crossRepel       = 0.05  // slightly higher so different groups keep soft boundaries

	damping = 0.93 // a touch less damping so they wiggle/relax instead of freezing
)

// Particle belongs to a population (group) and a bubble cluster (cid)
type Particle struct {
	x, y   float64
	vx, vy float64
	group  int // 0 = cyan foam, 1 = yellow foam
	cid    int // cluster id within its group
}

type Cluster struct {
	id       int
	group    int
	members  []*Particle
	pressure float64
}

var (
	parts    []*Particle
	clusters []*Cluster
)

// ---------- Utilities ----------

func randSym() float64 { return rand.Float64()*2 - 1 }

func angleBetween(p, a, b *Particle) float64 {
	v1x, v1y := a.x-p.x, a.y-p.y
	v2x, v2y := b.x-p.x, b.y-p.y
	dot := v1x*v2x + v1y*v2y
	n1 := math.Hypot(v1x, v1y)
	n2 := math.Hypot(v2x, v2y)
	if n1 == 0 || n2 == 0 {
		return 0
	}
	cos := dot / (n1 * n2)
	if cos < -1 {
		cos = -1
	} else if cos > 1 {
		cos = 1
	}
	return math.Acos(cos)
}

func wrap(x, max float64) float64 {
	if x < 0 {
		return x + max
	}
	if x >= max {
		return x - max
	}
	return x
}

// Minimum-image distance on a torus (wrap-around space)
func torusDelta(ax, ay, bx, by float64) (dx, dy float64) {
	dx = bx - ax
	dy = by - ay
	if dx > screenW/2 {
		dx -= screenW
	} else if dx < -screenW/2 {
		dx += screenW
	}
	if dy > screenH/2 {
		dy -= screenH
	} else if dy < -screenH/2 {
		dy += screenH
	}
	return
}

// ---------- Neighborhoods ----------

func neighborsSameGroup(p *Particle, cutoff float64) []*Particle {
	out := make([]*Particle, 0, 16)
	r2 := cutoff * cutoff
	for _, q := range parts {
		if q == p || q.group != p.group {
			continue
		}
		dx, dy := torusDelta(p.x, p.y, q.x, q.y)
		if dx*dx+dy*dy <= r2 {
			out = append(out, q)
		}
	}
	return out
}

func neighborsCrossGroup(p *Particle, cutoff float64) []*Particle {
	out := make([]*Particle, 0, 8)
	r2 := cutoff * cutoff
	for _, q := range parts {
		if q == p || q.group == p.group {
			continue
		}
		dx, dy := torusDelta(p.x, p.y, q.x, q.y)
		if dx*dx+dy*dy <= r2 {
			out = append(out, q)
		}
	}
	return out
}

// ---------- Cluster pressure ----------

func (c *Cluster) updatePressure() {
	n := len(c.members)
	if n <= 0 {
		c.pressure = 0
		return
	}
	c.pressure = pressureScale / float64(n)
}

// ---------- Dynamics ----------

func updateParticle(p *Particle, cl *Cluster) {
	// ---------- Metabolic / Brownian noise ----------
	// Small clusters jitter more than large ones
	kick := cl.pressure * noiseStrength
	p.vx += randSym() * kick
	p.vy += randSym() * kick

	// ---------- Cluster cohesion ----------
	if len(cl.members) > 0 {
		cx, cy := 0.0, 0.0
		for _, m := range cl.members {
			cx += m.x
			cy += m.y
		}
		cx /= float64(len(cl.members))
		cy /= float64(len(cl.members))
		dx, dy := torusDelta(p.x, p.y, cx, cy)
		// gentle pull toward cluster center
		p.vx += dx * 0.002
		p.vy += dy * 0.002
	}

	// ---------- Same-population neighbors ----------
	nSame := neighborsSameGroup(p, neighborRadiusSame)
	for _, n := range nSame {
		dx, dy := torusDelta(p.x, p.y, n.x, n.y)
		dist := math.Hypot(dx, dy)
		if dist == 0 {
			continue
		}
		diff := dist - preferredSpacing
		// soft spring (restoring force)
		f := baseTension * diff / dist
		p.vx += dx * f
		p.vy += dy * f
	}

	// ---------- Polygonal correction ----------
	if len(nSame) >= 2 {
		target := 2 * math.Pi / 3 // 120°
		for i := 0; i < len(nSame)-1; i++ {
			for j := i + 1; j < len(nSame); j++ {
				ang := angleBetween(p, nSame[i], nSame[j])
				delta := ang - target
				// pseudo-torque
				p.vx -= polygonBias * delta
				p.vy += polygonBias * delta
			}
		}
	}

	// ---------- Cross-population repulsion ----------
	nCross := neighborsCrossGroup(p, neighborRadiusCross)
	for _, n := range nCross {
		dx, dy := torusDelta(p.x, p.y, n.x, n.y)
		dist2 := dx*dx + dy*dy
		if dist2 == 0 {
			continue
		}
		dist := math.Sqrt(dist2)
		// soft 1/r repulsion
		F := crossRepel / (dist + 1e-6)
		p.vx -= (dx / dist) * F
		p.vy -= (dy / dist) * F
	}
}

func step() {
	// Refresh cluster membership
	for _, c := range clusters {
		c.members = c.members[:0]
	}
	for _, p := range parts {
		clusters[p.cid].members = append(clusters[p.cid].members, p)
	}
	for _, c := range clusters {
		c.updatePressure()
	}

	// Forces & integration
	for _, p := range parts {
		cl := clusters[p.cid]
		updateParticle(p, cl)

		p.x += p.vx
		p.y += p.vy
		p.vx *= damping
		p.vy *= damping

		// Toroidal wrap
		p.x = wrap(p.x, screenW)
		p.y = wrap(p.y, screenH)
	}
}

// ---------- Rendering (Canvas) ----------

func draw(cv *canvas.Canvas) {
	// Background
	cv.SetFillStyle("#000000")
	cv.FillRect(0, 0, float64(screenW), float64(screenH))

	// Draw particles as tiny squares (fast & simple)
	for _, p := range parts {
		if p.group == 0 {
			cv.SetFillStyle("#00DCFF") // cyan-ish
		} else {
			cv.SetFillStyle("#FFDC00") // yellow-ish
		}
		// center a 3x3 square at (x,y)
		cv.FillRect(p.x-1.5, p.y-1.5, 3, 3)
	}
}

// ---------- Initialization ----------

func newCluster(id, group int) *Cluster {
	return &Cluster{id: id, group: group, members: make([]*Particle, 0, numParts/8)}
}

func initSystem() {
	parts = make([]*Particle, 0, numParts)
	clusters = make([]*Cluster, 0, 10)

	// Cyan foam: 5 clusters
	for cid := 0; cid < 5; cid++ {
		c := newCluster(cid, 0)
		clusters = append(clusters, c)
		for i := 0; i < numParts/12; i++ {
			x := rand.Float64()*float64(screenW-100) + 50
			y := rand.Float64()*float64(screenH-100) + 50
			parts = append(parts, &Particle{x: x, y: y, group: 0, cid: cid})
		}
	}

	// Yellow foam: 5 clusters
	for cid := 5; cid < 10; cid++ {
		c := newCluster(cid, 1)
		clusters = append(clusters, c)
		for i := 0; i < numParts/12; i++ {
			x := rand.Float64()*float64(screenW-100) + 50
			y := rand.Float64()*float64(screenH-100) + 50
			parts = append(parts, &Particle{x: x, y: y, group: 1, cid: cid})
		}
	}
}

// ---------- Main ----------

func main() {
	rand.Seed(time.Now().UnixNano())

	wnd, cv, err := sdlcanvas.CreateWindow(screenW, screenH, "Foam (cyan vs yellow) - sdlcanvas")
	if err != nil {
		log.Fatal(err)
	}
	defer wnd.Destroy()

	initSystem()

	wnd.MainLoop(func() {
		step()
		draw(cv)
	})
}
