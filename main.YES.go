package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

const (
	width        = 2700
	height       = 1000
	padding      = 50
	particleSize = 5
)

type particle struct {
	x, y float64
	vx   float64
	vy   float64
	col  color.Color
}

var (
	particles []*particle
	wg        sync.WaitGroup
)

func randomX() float64 {
	return rand.Float64()*(width-padding*2) + padding
}

func randomY() float64 {
	return rand.Float64()*(height-padding*2) + padding
}

func create(number int, col color.Color) []particle {
	group := make([]particle, number)
	for i := 0; i < number; i++ {
		group[i] = particle{x: randomX(), y: randomY(), col: col}
		particles = append(particles, &group[i])
	}
	return group
}

func rule(particles1 []particle, particles2 []particle, g float64) {
	wg.Add(len(particles1))
	for i := 0; i < len(particles1); i++ {
		go func(i int) {
			defer wg.Done()
			a := &particles1[i]
			fx, fy := 0.0, 0.0

			for j := 0; j < len(particles2); j++ {
				b := &particles2[j]
				if a == b {
					continue
				}
				dx, dy := a.x-b.x, a.y-b.y
				d2 := dx*dx + dy*dy
				if d2 == 0 {
					continue
				}
				d := math.Sqrt(d2)

				if d < 120 {
					if d < 15 {
						F := -1.5 / d
						fx += F * dx
						fy += F * dy
					} else {
						F := g * (d - 40) / 120.0
						fx += F * dx / d
						fy += F * dy / d
					}
				}
			}

			drag := 0.85
			a.vx = a.vx*drag + fx*0.05 + (rand.Float64()-0.5)*0.02
			a.vy = a.vy*drag + fy*0.05 + (rand.Float64()-0.5)*0.02

			a.x += a.vx
			a.y += a.vy

			if a.x <= 0 || a.x >= float64(width-particleSize) {
				a.vx *= -0.7
			}
			if a.y <= 0 || a.y >= float64(height-particleSize) {
				a.vy *= -0.7
			}
		}(i)
	}
	wg.Wait()
}

// ---------------- Ebiten Game ------------------

type Game struct {
	start time.Time
	face  font.Face
}

func (g *Game) Update() error {
	startTime := time.Now()

	// Simulation rules
	rule(green, green, -0.32)
	rule(green, red, -0.17)
	rule(green, yellow, 0.34)
	rule(red, red, -0.1)
	rule(red, green, -0.34)
	rule(yellow, yellow, 0.15)
	rule(yellow, green, -0.20)

	elapsed := time.Since(startTime)
	g.start = elapsed
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	// Background
	screen.Fill(color.Black)

	// Draw particles
	for _, p := range particles {
		ebitenutil.DrawRect(screen, p.x, p.y, particleSize, particleSize, p.col)
	}

	// FPS text
	fpsText := fmt.Sprintf("FPS: %.0f", ebiten.ActualFPS())
	ebitenutil.DebugPrint(screen, fpsText)
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return width, height
}

// ---------------- Main ------------------

var (
	yellow []particle
	red    []particle
	green  []particle
)

func main() {
	// Load font (optional)
	tt, err := opentype.Parse(goregular_ttf) // replace goregular_ttf with your font []byte
	if err != nil {
		log.Fatal(err)
	}
	face, _ := opentype.NewFace(tt, &opentype.FaceOptions{
		Size:    24,
		DPI:     72,
		Hinting: font.HintingFull,
	})

	// Initialize groups
	yellow = create(2000, color.RGBA{255, 255, 0, 255})
	red = create(1000, color.RGBA{255, 0, 0, 255})
	green = create(2000, color.RGBA{0, 255, 0, 255})

	game := &Game{face: face}

	ebiten.SetWindowSize(1350, 500) // scaled window
	ebiten.SetWindowTitle("Artificial Life - Ebiten")
	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
