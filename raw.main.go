// lenia_ebiten_dynamic_speed.go
package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text"
	"golang.org/x/image/font/basicfont"
)

// ---------- Simulation parameters ----------
const (
	gridW    = 240
	gridH    = 160
	cellSize = 3
	radius   = 6.0
)

// ---------- Types ----------
type KernelEntry struct {
	dx, dy int
	w      float64
}

type Game struct {
	A        [][]float64
	Anext    [][]float64
	kernel   []KernelEntry
	Knorm    float64
	dt       float64
	mu       float64
	sigma    float64
	texture  *ebiten.Image
	frame    int
	start    time.Time
	lastFPS  int
	evoSpeed float64 // evolution speed factor (manual control)
}

// ---------- Utility ----------
func clamp(v, a, b float64) float64 {
	if v < a {
		return a
	}
	if v > b {
		return b
	}
	return v
}

func wrap(x, m int) int {
	if x >= 0 {
		return x % m
	}
	return (x%m + m) % m
}

// ---------- Kernel generation ----------
func buildKernel(R float64, shellSigma float64) ([]KernelEntry, float64) {
	var entries []KernelEntry
	Kc := func(rNorm float64) float64 {
		x := (rNorm - 0.5) / shellSigma
		return math.Exp(-0.5 * x * x)
	}

	Ri := int(math.Ceil(R))
	var sum float64
	for dy := -Ri; dy <= Ri; dy++ {
		for dx := -Ri; dx <= Ri; dx++ {
			dist := math.Hypot(float64(dx), float64(dy))
			if dist <= R {
				rnorm := dist / R * 2
				weight := Kc(rnorm)
				entries = append(entries, KernelEntry{dx: dx, dy: dy, w: weight})
				sum += weight
			}
		}
	}
	if sum == 0 {
		sum = 1
	}
	for i := range entries {
		entries[i].w /= sum
	}
	return entries, 1.0
}

// ---------- Growth mapping ----------
func growth(u, mu, sigma float64) float64 {
	if sigma <= 0 {
		return 0
	}
	val := 2*math.Exp(-((u-mu)*(u-mu))/(2*sigma*sigma)) - 1
	if val > 1 {
		val = 1
	} else if val < -1 {
		val = -1
	}
	return val
}

// ---------- Initialize ----------
func NewGame() *Game {
	rand.Seed(time.Now().UnixNano())
	A := make([][]float64, gridH)
	Anext := make([][]float64, gridH)
	for y := 0; y < gridH; y++ {
		A[y] = make([]float64, gridW)
		Anext[y] = make([]float64, gridW)
	}

	cx, cy := gridW/2, gridH/2
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			d := math.Hypot(float64(x-cx), float64(y-cy))
			if d < 16 {
				A[y][x] = 0.8 * math.Exp(-d*d/(2*8*8))
			}
			if rand.Float64() < 0.0015 {
				A[y][x] = rand.Float64()
			}
		}
	}

	shellSigma := 0.15
	kernel, knorm := buildKernel(radius, shellSigma)

	tex := ebiten.NewImage(gridW, gridH)
	g := &Game{
		A:        A,
		Anext:    Anext,
		kernel:   kernel,
		Knorm:    knorm,
		dt:       0.08,
		mu:       0.3,
		sigma:    0.06,
		texture:  tex,
		start:    time.Now(),
		evoSpeed: 1.0,
	}
	return g
}

// ---------- Update ----------
func (g *Game) step() {
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			var u float64
			for _, k := range g.kernel {
				nx := wrap(x+k.dx, gridW)
				ny := wrap(y+k.dy, gridH)
				u += k.w * g.A[ny][nx]
			}
			grow := growth(u, g.mu, g.sigma)
			val := g.A[y][x] + g.dt*grow
			g.Anext[y][x] = clamp(val, 0.0, 1.0)
		}
	}
	g.A, g.Anext = g.Anext, g.A
}

func (g *Game) Update() error {
	// Control evolution speed with keys
	if ebiten.IsKeyPressed(ebiten.KeyEqual) { // '=' (shift gives '+')
		g.evoSpeed *= 1.02
		if g.evoSpeed > 10 {
			g.evoSpeed = 10
		}
	}
	if ebiten.IsKeyPressed(ebiten.KeyMinus) { // '-' key
		g.evoSpeed *= 0.98
		if g.evoSpeed < 0.1 {
			g.evoSpeed = 0.1
		}
	}

	// Dynamic evolution of parameters
	t := float64(g.frame) * 0.002 * g.evoSpeed
	g.mu = 0.3 + 0.1*math.Sin(t*0.5)
	g.sigma = 0.06 + 0.02*math.Cos(t*0.3)
	g.dt = 0.07 + 0.02*math.Sin(t*0.7)

	shellSigma := 0.1 + 0.1*math.Abs(math.Sin(t*0.2))
	g.kernel, g.Knorm = buildKernel(radius, shellSigma)

	g.step()
	g.frame++
	if g.frame%30 == 0 {
		elapsed := time.Since(g.start).Seconds()
		if elapsed > 0 {
			g.lastFPS = int(float64(g.frame) / elapsed)
		}
	}
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			v := g.A[y][x]
			r, gg, b := colorRamp(v)
			g.texture.Set(x, y, color.NRGBA{R: r, G: gg, B: b, A: 0xFF})
		}
	}
	op := &ebiten.DrawImageOptions{}
	op.GeoM.Scale(float64(cellSize), float64(cellSize))
	op.Filter = ebiten.FilterNearest
	screen.DrawImage(g.texture, op)

	txt := fmt.Sprintf("μ: %.3f  σ: %.3f  Δt: %.3f  EvoSpeed: %.2f  FPS: %d",
		g.mu, g.sigma, g.dt, g.evoSpeed, g.lastFPS)
	text.Draw(screen, txt, basicfont.Face7x13, 6, 18, color.White)
	help := "Keys: [+] faster   [-] slower"
	text.Draw(screen, help, basicfont.Face7x13, 6, 34, color.White)
}

func (g *Game) Layout(outW, outH int) (int, int) {
	return gridW * cellSize, gridH * cellSize
}

// ---------- Color ramp ----------
func colorRamp(v float64) (r, g, b uint8) {
	v = clamp(v, 0, 1)
	if v < 0.5 {
		t := v / 0.5
		return uint8(20 + 50*t), uint8(50 + 150*t), uint8(200 - 100*t)
	}
	t := (v - 0.5) / 0.5
	return uint8(70 + 180*t), uint8(200 - 80*t), uint8(100 + 150*t)
}

// ---------- main ----------
func main() {
	ebiten.SetWindowSize(gridW*cellSize, gridH*cellSize)
	ebiten.SetWindowTitle("Dynamic Lenia-like Artificial Life (Manual Speed Control)")

	game := NewGame()

	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
