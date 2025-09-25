// lenia_yeast.go
package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text"
	"golang.org/x/image/font/basicfont"
)

// ---------- Simulation parameters ----------
const (
	gridW        = 300
	gridH        = 300
	cellSize     = 3
	evalSteps    = 800
	populationSz = 30
	elitism      = 5
	mutationRate = 0.6

	// nutrient rules
	initialNutrient = 0.9    // initial nutrient concentration
	consumptionRate = 0.005  // how fast cells consume nutrients
	diffusionRate   = 0.1    // how fast nutrients spread
	replenishRate   = 0.0003 // small global replenishment

	// budding rules
	budThreshold   = 0.75
	budChance      = 0.002
	budNutrientMin = 0.4 // min nutrient level for budding
	budRadius      = 6
)

// ---------- Types ----------
type KernelEntry struct {
	dx, dy int
	w      float64
}

type Genome struct {
	Mu, Sigma, Radius, ShellSigma, Dt, ColorBias float64
	Fitness                                      float64
}

type Game struct {
	A       [][]float64 // cell density
	Anext   [][]float64
	N       [][]float64 // nutrient grid
	Nnext   [][]float64
	kernel  []KernelEntry
	Knorm   float64
	texture *ebiten.Image

	// runtime
	generation      int
	population      []Genome
	currentIndex    int
	stepCount       int
	autoEvolve      bool
	autoEvolveDelay time.Duration
	lastEvolveTime  time.Time
	frame           int
	start           time.Time
	lastFPS         int
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

// ---------- Kernel ----------
func buildKernel(R float64, shellSigma float64) ([]KernelEntry, float64) {
	var entries []KernelEntry
	if R <= 0 {
		R = 1
	}
	if shellSigma <= 0 {
		shellSigma = 0.15
	}
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

func growth(u, mu, sigma float64) float64 {
	if sigma <= 0 {
		return 0
	}
	val := 2*math.Exp(-((u-mu)*(u-mu))/(2*sigma*sigma)) - 1
	return clamp(val, -1, 1)
}

// ---------- Initialization ----------
func NewGame() *Game {
	rand.Seed(time.Now().UnixNano())
	A := make([][]float64, gridH)
	Anext := make([][]float64, gridH)
	N := make([][]float64, gridH)
	Nnext := make([][]float64, gridH)
	for y := 0; y < gridH; y++ {
		A[y] = make([]float64, gridW)
		Anext[y] = make([]float64, gridW)
		N[y] = make([]float64, gridW)
		Nnext[y] = make([]float64, gridW)
		for x := 0; x < gridW; x++ {
			N[y][x] = initialNutrient
		}
	}

	g := &Game{
		A:               A,
		Anext:           Anext,
		N:               N,
		Nnext:           Nnext,
		texture:         ebiten.NewImage(gridW, gridH),
		autoEvolveDelay: 3 * time.Second,
		lastEvolveTime:  time.Now(),
		start:           time.Now(),
	}
	// init population
	g.population = make([]Genome, populationSz)
	for i := 0; i < populationSz; i++ {
		g.population[i] = randomGenome()
	}
	g.applyGenomeKernel(&g.population[0])
	g.seedFromGenome(&g.population[0])
	return g
}

func randomGenome() Genome {
	return Genome{
		Mu:         0.18 + rand.Float64()*0.5,
		Sigma:      0.02 + rand.Float64()*0.18,
		Radius:     3.0 + rand.Float64()*8.0,
		ShellSigma: 0.08 + rand.Float64()*0.3,
		Dt:         0.03 + rand.Float64()*0.12,
		ColorBias:  rand.Float64()*1.0 - 0.5,
	}
}

func (g *Game) applyGenomeKernel(gen *Genome) {
	k, kn := buildKernel(gen.Radius, gen.ShellSigma)
	g.kernel = k
	g.Knorm = kn
}

func (g *Game) seedFromGenome(gen *Genome) {
	cx, cy := gridW/2, gridH/2
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			g.A[y][x] = 0
			g.Anext[y][x] = 0
		}
	}
	base := int(math.Max(6, gen.Radius*1.5))
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			d := math.Hypot(float64(x-cx), float64(y-cy))
			if d < float64(base) {
				g.A[y][x] = 0.6 * math.Exp(-d*d/(2*float64(base)*float64(base)))
			}
		}
	}
}

// ---------- Simulation ----------
func (g *Game) step(gen *Genome) {
	// cell update
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			var u float64
			for _, k := range g.kernel {
				nx := wrap(x+k.dx, gridW)
				ny := wrap(y+k.dy, gridH)
				u += k.w * g.A[ny][nx]
			}
			grow := growth(u, gen.Mu, gen.Sigma)

			// nutrient consumption
			consumed := consumptionRate * g.A[y][x]
			if g.N[y][x] > consumed {
				g.N[y][x] -= consumed
				val := g.A[y][x] + gen.Dt*grow
				g.Anext[y][x] = clamp(val, 0.0, 1.0)
			} else {
				g.Anext[y][x] = g.A[y][x] * 0.98 // starve slowly
			}
		}
	}

	// nutrient diffusion + replenish
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			avg := g.N[y][x]
			count := 1.0
			for _, k := range g.kernel {
				nx := wrap(x+k.dx, gridW)
				ny := wrap(y+k.dy, gridH)
				avg += g.N[ny][nx]
				count++
			}
			avg /= count
			// diffusion
			g.Nnext[y][x] = g.N[y][x] + diffusionRate*(avg-g.N[y][x])
			// replenishment
			g.Nnext[y][x] = clamp(g.Nnext[y][x]+replenishRate, 0, 1)
		}
	}

	// swap buffers
	g.A, g.Anext = g.Anext, g.A
	g.N, g.Nnext = g.Nnext, g.N
}

// ---------- Budding reproduction ----------
func (g *Game) reproduceBudding() {
	for y := 1; y < gridH-1; y++ {
		for x := 1; x < gridW-1; x++ {
			if g.A[y][x] > budThreshold && g.N[y][x] > budNutrientMin && rand.Float64() < budChance {
				dx := rand.Intn(3) - 1
				dy := rand.Intn(3) - 1
				bx := wrap(x+dx*budRadius, gridW)
				by := wrap(y+dy*budRadius, gridH)
				for yy := -budRadius; yy <= budRadius; yy++ {
					for xx := -budRadius; xx <= budRadius; xx++ {
						dist := math.Hypot(float64(xx), float64(yy))
						if dist < float64(budRadius) {
							g.A[wrap(by+yy, gridH)][wrap(bx+xx, gridW)] +=
								0.6 * math.Exp(-dist*dist/(2*float64(budRadius)*float64(budRadius)))
						}
					}
				}
			}
		}
	}
}

// ---------- Evolution (shortened for demo) ----------
func (g *Game) evaluateGenome(gen *Genome) float64 {
	g.applyGenomeKernel(gen)
	g.seedFromGenome(gen)
	// reset nutrients
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			g.N[y][x] = initialNutrient
		}
	}
	var total float64
	for step := 0; step < evalSteps; step++ {
		g.step(gen)
		if step%5 == 0 {
			sum := 0.0
			for y := 0; y < gridH; y++ {
				for x := 0; x < gridW; x++ {
					sum += g.A[y][x]
				}
			}
			total += sum
		}
	}
	return total / float64(evalSteps)
}

func (g *Game) evolveOnce() {
	for i := range g.population {
		score := g.evaluateGenome(&g.population[i])
		g.population[i].Fitness = score
	}
	sort.Slice(g.population, func(i, j int) bool {
		return g.population[i].Fitness > g.population[j].Fitness
	})
	g.generation++
	g.currentIndex = 0
	g.applyGenomeKernel(&g.population[0])
	g.seedFromGenome(&g.population[0])
}

// ---------- Ebiten ----------
func (g *Game) Update() error {
	cur := &g.population[g.currentIndex]
	g.step(cur)
	g.reproduceBudding()

	// FPS tracking
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
	bias := g.population[g.currentIndex].ColorBias
	for y := 0; y < gridH; y++ {
		for x := 0; x < gridW; x++ {
			v := clamp(g.A[y][x]+bias*0.08, 0, 1)
			n := g.N[y][x]
			r, gg, b := colorRamp(v, n)
			g.texture.Set(x, y, color.NRGBA{R: r, G: gg, B: b, A: 0xFF})
		}
	}
	op := &ebiten.DrawImageOptions{}
	op.GeoM.Scale(float64(cellSize), float64(cellSize))
	screen.DrawImage(g.texture, op)
	txt := fmt.Sprintf("Gen:%d  Fitness:%.3f  FPS:%d", g.generation, g.population[0].Fitness, g.lastFPS)
	text.Draw(screen, txt, basicfont.Face7x13, 6, 16, color.White)
}

func (g *Game) Layout(outW, outH int) (int, int) {
	return gridW * cellSize, gridH * cellSize
}

// ---------- Color mapping ----------
func colorRamp(v, nutrient float64) (r, g, b uint8) {
	// mix: blue for nutrient, green/yellow for cells
	v = clamp(v, 0, 1)
	n := clamp(nutrient, 0, 1)
	return uint8(50 + 205*v), uint8(50 + 150*n), uint8(200 - 120*v)
}

// ---------- main ----------
func main() {
	ebiten.SetWindowSize(gridW*cellSize, gridH*cellSize)
	ebiten.SetWindowTitle("Artificial Yeast with Nutrient-Based Budding")
	game := NewGame()
	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
