package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text"
	"golang.org/x/image/font/basicfont"
)

// ---------- Simulation parameters (tweak these) ----------
const (
	gridW      = 400        // lattice width
	gridH      = 400        // lattice height
	cellSize   = 3          // display pixel size for each lattice cell
	radius     = 6.0        // neighborhood radius in grid units (R)
	dtDefault  = 0.1618033  // Δt
	muDefault  = 0.01618033 // μ for growth mapping
	sigDefault = 0.01618033 // σ for growth mapping
)

// The number of goroutines to use for parallel processing.
// A common choice is runtime.NumCPU() or a slightly smaller number.
var numWorkers = runtime.NumCPU()

// ---------- Types ----------
type KernelEntry struct {
	dx, dy int
	w      float64
}

type Game struct {
	A       [][]float64 // current state grid [y][x]
	Anext   [][]float64 // next state grid
	kernel  []KernelEntry
	Knorm   float64
	dt      float64
	mu      float64
	sigma   float64
	texture *ebiten.Image // gridW x gridH image we write pixels into and scale up
	frame   int
	start   time.Time
	lastFPS int
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
	// positive remainder
	return (x%m + m) % m
}

// ---------- Kernel generation (Unchanged) ----------
func buildKernel(R float64) ([]KernelEntry, float64) {
	var entries []KernelEntry
	shellSigma := 0.1618033
	Kc := func(rNorm float64) float64 {
		x := (rNorm - 0.5) / shellSigma
		return math.Exp(-0.5 * x * x)
	}

	Ri := int(math.Ceil(R))
	var sum float64
	for dy := -Ri; dy <= Ri; dy++ {
		for dx := -Ri; dx <= Ri; dx++ {
			dxF := float64(dx)
			dyF := float64(dy)
			dist := math.Hypot(dxF, dyF)
			if dist <= R {
				rnorm := dist / R
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

// ---------- Growth mapping (Unchanged) ----------
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

// ---------- Initialize (Unchanged) ----------
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
			A[y][x] = 0.0
			if d < 16 {
				A[y][x] = 0.8 * math.Exp(-d*d/(2*8*8))
			}
			if rand.Float64() < 0.001 {
				A[y][x] = rand.Float64()*0.8 + 0.1
			}
		}
	}

	kernel, knorm := buildKernel(radius)

	tex := ebiten.NewImage(gridW, gridH)

	g := &Game{
		A:       A,
		Anext:   Anext,
		kernel:  kernel,
		Knorm:   knorm,
		dt:      dtDefault,
		mu:      muDefault,
		sigma:   sigDefault,
		texture: tex,
		start:   time.Now(),
	}
	return g
}

// ---------- Parallel Update step: compute U = K * A, then growth, then update Anext ----------

// workerStep calculates the next state for a subset of the rows (from startY to endY-1)
func (g *Game) workerStep(startY, endY int, wg *sync.WaitGroup) {
	defer wg.Done()

	// Cache frequently accessed fields to reduce struct dereferencing within the hot loops
	A := g.A
	Anext := g.Anext
	kernel := g.kernel
	dt := g.dt
	mu := g.mu
	sigma := g.sigma

	for y := startY; y < endY; y++ {
		for x := 0; x < gridW; x++ {
			var u float64
			// convolution sum
			for _, k := range kernel {
				nx := wrap(x+k.dx, gridW)
				ny := wrap(y+k.dy, gridH)
				// Accessing A[ny][nx] is safe because we are only writing to Anext
				u += k.w * A[ny][nx]
			}
			// growth mapping
			grow := growth(u, mu, sigma)
			// update
			val := A[y][x] + dt*grow
			// clip to [0,1]
			Anext[y][x] = clamp(val, 0.0, 1.0)
		}
	}
}

func (g *Game) parallelStep() {
	var wg sync.WaitGroup
	rowsPerWorker := gridH / numWorkers

	// Launch workers
	for i := 0; i < numWorkers; i++ {
		startY := i * rowsPerWorker
		endY := (i + 1) * rowsPerWorker

		// Ensure the last worker handles any remainder rows
		if i == numWorkers-1 {
			endY = gridH
		}

		wg.Add(1)
		go g.workerStep(startY, endY, &wg)
	}

	// Wait for all workers to finish
	wg.Wait()

	// swap
	g.A, g.Anext = g.Anext, g.A
}

// ---------- Ebiten game interface (Modified to use parallelStep) ----------
func (g *Game) Update() error {
	// keyboard controls for parameters (optional) - UNCHANGED
	if ebiten.IsKeyPressed(ebiten.KeyU) { // increase mu
		g.mu += 0.002
	}
	if ebiten.IsKeyPressed(ebiten.KeyJ) { // decrease mu
		g.mu -= 0.002
		if g.mu < 0 {
			g.mu = 0
		}
	}
	if ebiten.IsKeyPressed(ebiten.KeyI) { // increase sigma
		g.sigma += 0.001
	}
	if ebiten.IsKeyPressed(ebiten.KeyK) { // decrease sigma
		g.sigma -= 0.001
		if g.sigma < 0.0001 {
			g.sigma = 0.0001
		}
	}
	if ebiten.IsKeyPressed(ebiten.KeyO) { // increase dt
		g.dt += 0.001
	}
	if ebiten.IsKeyPressed(ebiten.KeyL) { // decrease dt
		g.dt -= 0.001
		if g.dt < 0.001 {
			g.dt = 0.001
		}
	}

	// Use the parallelized step function
	stepsPerFrame := 1
	for i := 0; i < stepsPerFrame; i++ {
		g.parallelStep() // *** KEY CHANGE HERE ***
	}

	g.frame++
	// FPS estimate every ~30 frames
	if g.frame%30 == 0 {
		elapsed := time.Since(g.start).Seconds()
		if elapsed > 0 {
			g.lastFPS = int(float64(g.frame) / elapsed)
		}
	}
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	// write A into texture (gridW x gridH) as colored pixels
	// Ebiten's Image.Set (and ReplacePixels) can be slow.
	// For small grids like this, it might be fine, but for performance,
	// a common Ebiten optimization is to use ebiten.Image.WritePixels
	// on a single large byte array, which is faster than calling Set
	// in a loop. I'll stick to Set for simplicity as the parallel
	// step is the main bottleneck.

	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			v := g.A[y][x]
			r, gg, b := colorRamp(v)
			g.texture.Set(x, y, color.NRGBA{R: r, G: gg, B: b, A: 0xFF})
		}
	}
	// draw scaled to window
	op := &ebiten.DrawImageOptions{}
	op.GeoM.Scale(float64(cellSize), float64(cellSize))
	op.Filter = ebiten.FilterNearest
	screen.DrawImage(g.texture, op)

	// overlay text for parameters and instructions
	txt := fmt.Sprintf("μ: %.3f  σ: %.3f  Δt: %.3f  R: %.1f    FPS(est): %d | Workers: %d", g.mu, g.sigma, g.dt, radius, g.lastFPS, numWorkers)
	text.Draw(screen, txt, basicfont.Face7x13, 6, 18, color.White)

	help := "Keys: U/J μ+/-   I/K σ+/-   O/L Δt+/-   (wrap boundary, gaussian shell, growth=gaussian)"
	text.Draw(screen, help, basicfont.Face7x13, 6, 34, color.White)
}

func (g *Game) Layout(outW, outH int) (int, int) {
	return gridW * cellSize, gridH * cellSize
}

// ---------- simple color ramp mapping (Unchanged) ----------
func colorRamp(v float64) (r, g, b uint8) {
	v = clamp(v, 0, 1)
	if v < 0.5 {
		t := v / 0.5
		// The math here uses fixed constants and float ops, which is fine
		return uint8(20 + 50*t), uint8(50 + 150*t), uint8(200 - 100*t)
	}
	t := (v - 0.5) / 0.5
	return uint8(70 + 180*t), uint8(200 - 80*t), uint8(100 + 150*t)
}

// ---------- main (Modified to set GOMAXPROCS) ----------
func main() {
	// Set the number of operating system threads that can execute user-level Go code simultaneously.
	// This is the default behavior with Go 1.5+, but explicitly setting it to numWorkers (or NumCPU())
	// ensures we are using all available cores for the goroutines.
	runtime.GOMAXPROCS(numWorkers)

	ebiten.SetWindowSize(gridW*cellSize, gridH*cellSize)
	ebiten.SetWindowTitle(fmt.Sprintf("Lenia-like Artificial Cell (Ebiten, Parallel: %d Workers)", numWorkers))

	game := NewGame()

	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
