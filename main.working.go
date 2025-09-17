package main

import (
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/tfriedel6/canvas"
	"github.com/tfriedel6/canvas/sdlcanvas"
)

// ===============================
// PARAMETERS
// ===============================
const (
	screenW  = 1200
	screenH  = 800
	numParts = 600 // total particles

	// Neighborhood radii
	neighborRadiusSame  = 22.0 // same-population neighbor search
	neighborRadiusCross = 18.0 // cross-population (interface) radius

	// LIFE-LIKE PRESET
	preferredSpacing = 14.0  // a bit tighter packing than default
	baseTension      = 0.028 // stronger membranes → smooth, rounded cells
	polygonBias      = 0.020 // pushes toward ~120° → honeycomb-ish, tissue-like
	noiseStrength    = 0.12  // gentle Brownian jiggle (not too twitchy)
	crossRepel       = 0.03  // different types can touch/mix like real tissues

	damping = 0.95 // velocity damping
)

// ===============================
// PARTICLE STRUCT
// ===============================
type Particle struct {
	x, y   float64
	vx, vy float64
	typ    int
}

// ===============================
// INIT PARTICLES
// ===============================
func newParticles() []*Particle {
	parts := make([]*Particle, numParts)
	for i := 0; i < numParts; i++ {
		parts[i] = &Particle{
			x:   rand.Float64() * screenW,
			y:   rand.Float64() * screenH,
			vx:  0,
			vy:  0,
			typ: i % 2, // two populations
		}
	}
	return parts
}

// ===============================
// UPDATE LOOP
// ===============================
func updateParticles(parts []*Particle) {
	for i, p := range parts {
		fx, fy := 0.0, 0.0

		for j, q := range parts {
			if i == j {
				continue
			}

			dx := q.x - p.x
			dy := q.y - p.y
			dist := math.Hypot(dx, dy)
			if dist < 0.001 {
				continue
			}

			// normalize
			nx := dx / dist
			ny := dy / dist

			// same-type neighborhood
			if p.typ == q.typ && dist < neighborRadiusSame {
				// spring-like attraction/repulsion
				force := baseTension * (dist - preferredSpacing)

				// polygon bias → honeycomb-ish
				angleBias := math.Cos(3*math.Atan2(dy, dx)) * polygonBias
				force += angleBias

				fx += force * nx
				fy += force * ny
			}

			// cross-type interaction
			if p.typ != q.typ && dist < neighborRadiusCross {
				// gentle repulsion → keeps tissues separate but still mixing
				force := crossRepel * (neighborRadiusCross - dist)
				fx -= force * nx
				fy -= force * ny
			}
		}

		// Brownian / thermal jitter
		fx += (rand.Float64()*2 - 1) * noiseStrength
		fy += (rand.Float64()*2 - 1) * noiseStrength

		// integrate velocity
		p.vx = (p.vx + fx) * damping
		p.vy = (p.vy + fy) * damping

		// apply movement
		p.x += p.vx
		p.y += p.vy

		// wrap around boundaries
		if p.x < 0 {
			p.x += screenW
		}
		if p.x >= screenW {
			p.x -= screenW
		}
		if p.y < 0 {
			p.y += screenH
		}
		if p.y >= screenH {
			p.y -= screenH
		}
	}
}

// ===============================
// DRAW
func draw(c *canvas.Canvas, parts []*Particle) {
	// background
	c.SetFillStyle("#000000")
	c.FillRect(0, 0, float64(screenW), float64(screenH))

	for _, p := range parts {
		if p.typ == 0 {
			c.SetFillStyle("#66bb6a") // green-ish
		} else {
			c.SetFillStyle("#42a5f5") // blue-ish
		}
		c.BeginPath()
		c.Arc(p.x, p.y, 1.618033, 0, math.Pi*2, false) // full circle
		c.Fill()
	}
}

// ===============================
// MAIN
// ===============================
func main() {
	rand.Seed(time.Now().UnixNano())

	win, cv, err := sdlcanvas.CreateWindow(screenW, screenH, "Life-like Electro-Quantum Simulation")
	if err != nil {
		log.Fatal(err)
	}
	defer win.Destroy()

	parts := newParticles()

	win.MainLoop(func() {
		updateParticles(parts)
		draw(cv, parts)
	})
}
