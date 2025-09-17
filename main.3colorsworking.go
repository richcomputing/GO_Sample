package main

import (
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

// ===============================
// PARAMETERS
// ===============================
const (
	screenW  = 1200
	screenH  = 800
	numParts = 600

	neighborRadiusSame  = 22.0
	neighborRadiusCross = 18.0

	preferredSpacing = 14.0
	baseTension      = 0.028
	polygonBias      = 0.020
	noiseStrength    = 0.12
	crossRepel       = 0.03

	damping = 0.95
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
// GAME STRUCT
// ===============================
type Game struct {
	parts []*Particle
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
			typ: i % 2,
		}
	}
	return parts
}

// ===============================
// UPDATE LOOP
// ===============================
func (g *Game) Update() error {
	for i, p := range g.parts {
		fx, fy := 0.0, 0.0
		for j, q := range g.parts {
			if i == j {
				continue
			}
			dx := q.x - p.x
			dy := q.y - p.y
			dist := math.Hypot(dx, dy)
			if dist < 0.001 {
				continue
			}
			nx, ny := dx/dist, dy/dist

			if p.typ == q.typ && dist < neighborRadiusSame {
				force := baseTension * (dist - preferredSpacing)
				angleBias := math.Cos(3*math.Atan2(dy, dx)) * polygonBias
				force += angleBias
				fx += force * nx
				fy += force * ny
			}
			if p.typ != q.typ && dist < neighborRadiusCross {
				force := crossRepel * (neighborRadiusCross - dist)
				fx -= force * nx
				fy -= force * ny
			}
		}
		fx += (rand.Float64()*2 - 1) * noiseStrength
		fy += (rand.Float64()*2 - 1) * noiseStrength

		p.vx = (p.vx + fx) * damping
		p.vy = (p.vy + fy) * damping
		p.x += p.vx
		p.y += p.vy

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
	return nil
}

// ===============================
// DRAW
// ===============================
func (g *Game) Draw(screen *ebiten.Image) {
	screen.Fill(makeColor(0, 0, 0))
	for _, p := range g.parts {
		if p.typ == 0 {
			ebitenutil.DrawRect(screen, p.x, p.y, 2, 2, makeColor(102,
