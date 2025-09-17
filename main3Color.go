package main

import (
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"image/color"
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
	preferredSpacing = 14.0
	baseTension      = 0.028
	polygonBias      = 0.020
	noiseStrength    = 0.12
	crossRepel       = 0.03

	damping      = 0.95
	particleSize = 2.0 // radius
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
			typ: i % 3, // three populations
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
			nx := dx / dist
			ny := dy / dist

			// same-type neighborhood
			if p.typ == q.typ && dist < neighborRadiusSame {
				force := baseTension * (dist - preferredSpacing)
				angleBias := math.Cos(3*math.Atan2(dy, dx)) * polygonBias
				force += angleBias
				fx += force * nx
				fy += force * ny
			}

			// cross-type interaction
			if p.typ != q.typ && dist < neighborRadiusCross {
				force := crossRepel * (neighborRadiusCross - dist)
				if p.typ == 2 {
					force *= 1.8
				}
				fx -= force * nx
				fy -= force * ny
			}
		}

		// Brownian jitter
		fx += (rand.Float64()*2 - 1) * noiseStrength
		fy += (rand.Float64()*2 - 1) * noiseStrength

		// integrate velocity
		p.vx = (p.vx + fx) * damping
		p.vy = (p.vy + fy) * damping

		// apply movement
		p.x += p.vx
		p.y += p.vy

		// wrap around
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
// GAME LOOP
// ===============================
type Game struct {
	parts []*Particle
}

func (g *Game) Update() error {
	updateParticles(g.parts)
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	// Clear background
	screen.Fill(color.Black)

	// Draw particles
	for _, p := range g.parts {
		var col color.Color
		switch p.typ {
		case 0:
			col = color.RGBA{102, 187, 106, 255} // green
		case 1:
			col = color.RGBA{66, 165, 245, 255} // blue
		case 2:
			col = color.RGBA{239, 83, 80, 255} // red
		}
		ebitenutil.DrawRect(screen, p.x-particleSize/2, p.y-particleSize/2, particleSize, particleSize, col)
	}
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return screenW, screenH
}

// ===============================
// MAIN
// ===============================
func main() {
	rand.Seed(time.Now().UnixNano())
	game := &Game{parts: newParticles()}

	ebiten.SetWindowSize(screenW, screenH)
	ebiten.SetWindowTitle("Life-like Electro-Quantum Simulation (Ebiten)")

	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
