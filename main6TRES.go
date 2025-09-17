package main

import (
	"image/color"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

const (
	screenW, screenH = 1000, 800
	actorSize        = 6.0 // change this to resize all creatures
)

type Creature struct {
	x, y   float64
	energy float64
	kind   string // "cain" / "abel" / "mutant"
	tribe  int    // tribe ID
}

type Tribe struct {
	id   int
	col  color.RGBA
	name string
}

type Game struct {
	creatures []*Creature
	tribes    []*Tribe
	tick      int
	nextTribe int
}

func (g *Game) Update() error {
	g.tick++

	// ROTAS extinction every ~2000 ticks
	if g.tick%2000 == 0 {
		newCreatures := []*Creature{}
		for _, c := range g.creatures {
			if rand.Float64() < 0.2 { // 20% fossil rebirth
				c.energy = 5
				newCreatures = append(newCreatures, c)
			}
		}
		g.creatures = newCreatures
	}

	// Mouse click spawns a new tribe
	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		x, y := ebiten.CursorPosition()
		g.spawnTribe(float64(x), float64(y))
	}

	for _, c := range g.creatures {
		// random wandering
		c.x += rand.Float64()*4 - 2
		c.y += rand.Float64()*4 - 2
		if c.x < 0 {
			c.x = screenW
		}
		if c.x > screenW {
			c.x = 0
		}
		if c.y < 0 {
			c.y = screenH
		}
		if c.y > screenH {
			c.y = 0
		}

		// energy drain
		c.energy -= 0.01

		if c.kind == "cain" { // predator hunts prey
			for _, prey := range g.creatures {
				if prey.kind == "abel" && prey.energy > 0 {
					dx := prey.x - c.x
					dy := prey.y - c.y
					dist := math.Hypot(dx, dy)
					if dist < actorSize*2 {
						c.energy += 5
						prey.energy = -1 // prey dies
					}
				}
			}
		} else if c.kind == "abel" { // prey reproduces
			if rand.Float64() < 0.001 {
				kind := "abel"
				if rand.Float64() < 0.05 { // 5% chance mutation
					kind = "mutant"
				}
				g.creatures = append(g.creatures, &Creature{
					x:      c.x + rand.Float64()*10 - 5,
					y:      c.y + rand.Float64()*10 - 5,
					energy: 3,
					kind:   kind,
					tribe:  c.tribe,
				})
			}
		}
	}

	// Cull dead
	survivors := []*Creature{}
	for _, c := range g.creatures {
		if c.energy > 0 {
			survivors = append(survivors, c)
		}
	}
	g.creatures = survivors

	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.Fill(color.RGBA{0, 0, 0, 255})

	for _, c := range g.creatures {
		tribeCol := g.tribes[c.tribe].col
		var col color.RGBA

		switch c.kind {
		case "abel":
			col = tribeCol
		case "cain":
			col = color.RGBA{R: tribeCol.R / 2, G: tribeCol.G / 2, B: tribeCol.B / 2, A: 255}
		case "mutant":
			col = color.RGBA{R: 255, G: 255, B: 0, A: 255} // yellow = mutants
		}

		switch c.kind {
		case "abel": // Circle
			vector.DrawFilledCircle(screen, float32(c.x), float32(c.y), float32(actorSize), col, false)
		case "cain": // Triangle
			size := float32(actorSize)
			path := &vector.Path{}
			path.MoveTo(float32(c.x), float32(c.y)-size)
			path.LineTo(float32(c.x)-size*0.866, float32(c.y)+size/2)
			path.LineTo(float32(c.x)+size*0.866, float32(c.y)+size/2)
			path.Close()
			var vertices []ebiten.Vertex
			var indices []uint16
			vertices, indices = path.AppendVerticesAndIndicesForFilling(vertices, indices)
			// Create a 1x1 white image to use as texture
			img := ebiten.NewImage(1, 1)
			img.Fill(col)
			// Draw the filled triangle
			screen.DrawTriangles(vertices, indices, img, nil)

		}
	}
}

func (g *Game) Layout(outW, outH int) (int, int) {
	return screenW, screenH
}

func (g *Game) spawnTribe(x, y float64) {
	id := g.nextTribe
	g.nextTribe++

	// random tribe color
	tribeColor := color.RGBA{
		uint8(rand.Intn(200) + 55),
		uint8(rand.Intn(200) + 55),
		uint8(rand.Intn(200) + 55),
		255,
	}
	tribe := &Tribe{id: id, col: tribeColor, name: "Tribe"}
	g.tribes = append(g.tribes, tribe)

	// spawn initial Abel (prey)
	for i := 0; i < 30; i++ {
		g.creatures = append(g.creatures, &Creature{
			x:      x + rand.Float64()*40 - 20,
			y:      y + rand.Float64()*40 - 20,
			energy: 5,
			kind:   "abel",
			tribe:  id,
		})
	}
	// spawn initial Cain (predators)
	for i := 0; i < 8; i++ {
		g.creatures = append(g.creatures, &Creature{
			x:      x + rand.Float64()*40 - 20,
			y:      y + rand.Float64()*40 - 20,
			energy: 10,
			kind:   "cain",
			tribe:  id,
		})
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())
	game := &Game{}

	ebiten.SetWindowSize(screenW, screenH)
	ebiten.SetWindowTitle("Cyberlife: Tribes, Cain & Abel, ROTAS, Identity Symbols")
	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
