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
	preferredSpacing = 14.0
	baseTension      = 0.028
	polygonBias      = 0.020
	noiseStrength    = 0.12
	crossRepel       = 0.03

	damping      = 0.95
	particleSize = 1.618033 // <<-- adjust particle radius here
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
			typ: i % 3, // now three populations: 0,1,2
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
				force := baseTension * (dist - preferredSpacing)
				angleBias := math.Cos(3*math.Atan2(dy, dx)) * polygonBias
				force += angleBias
				fx += force * nx
				fy += force * ny
			}

			// cross-type interaction
			if p.typ != q.typ && dist < neighborRadiusCross {
				force := crossRepel * (neighborRadiusCross - dist)

				// make red aggressive particles (typ==2) push harder
				if p.typ == 2 {
					force *= 1.8
				}

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
// ===============================
func draw(c *canvas.Canvas, parts []*Particle) {
	// background
	c.SetFillStyle("#000000")
	c.FillRect(0, 0, float64(screenW), float64(screenH))

	for _, p := range parts {
		switch p.typ {
		case 0: // green circles
			c.SetFillStyle("#66bb6a")
			c.BeginPath()
			c.Arc(p.x, p.y, particleSize, 0, math.Pi*2, false)
			c.Fill()

		case 1: // blue circles
			c.SetFillStyle("#42a5f5")
			c.BeginPath()
			c.Arc(p.x, p.y, particleSize, 0, math.Pi*2, false)
			c.Fill()

		case 2: // red aggressive triangles
			c.SetFillStyle("#ef5350")
			c.BeginPath()
			size := particleSize * 1.5
			c.MoveTo(p.x, p.y-size)
			c.LineTo(p.x-size*0.866, p.y+size/2)
			c.LineTo(p.x+size*0.866, p.y+size/2)
			c.ClosePath()
			c.Fill()
		}
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


**********************

Julio Agosto 


Feb
************
Infrastructure Optimization and Cost Reduction:
- Deleted 97 unused Horizon Dedicated VMs.

- Deleted 2 AppStream fleets in Europe, only 2 left on Europe. 7 left in the US. We started with~44

- Worked with Cyber Workforce Identity Protection to delete 20 Applications from Azure.

- Created scripts to get closer to full automation for traditional VM creation. Isaac did a great job on this, and it will allow us to migrate the VMs out of GSC more efficiently.

Major Accomplishments: 
- Datadog Dashboard and alerts completed for VDI Monitoring.

- Built out Windows 11 Instant Clone Desktop Pool for EU, currently testing internally.

- Migrated VDI to new storage clusters.

- Migrated SEED Team to 39 new Dedicated Horizon VMs.

- Identified 24 VMs at GSC  that can be deleted and/or migrated to the 247/8 DCs. We started with 147 total, 74 were deleted and ~24 more  to be deleted in March leaving us with ~61 Windows VMs.

- Worked with Omnissa to rectify the Horizon sync issue so that we can move forward with the upgrade to 2409. 

Security Enhancements:
- Modified Zscaler in the US-based Horizon VDI environment to load faster to improve the user experience.
- Performed Windows patching.
- Updated Chrome and Edge browsers to address critical vulnerabilities.

- Updated 7-Zip to 2409 or later to address critical vulnerabilities

Training and Development:
- VM Template management for the Team

- Fully trained Chathurya for supporting the Horizon VDI environment and managing the App Volumes sector of Horizon. She is supporting the US off shift. *******************************************

MARCH**
********
Infrastructure Optimization and Cost Reduction:

After the significant challenges in the Vcenter, we upgraded the VDI environment to version 2406.  

Migrate DVMs, VMs and IC Pools to the new storage cluster - DC4_PMAX_EUT_CL03

Setup policy in order to allow the use of time zone mapping to match users location.

 Update DEM to change the Adobe EXE to the the Nitro EXE

Major Accomplishments:

Published MS Office via AppVols to EU Windows 11 Instant Clone Desktop Pool. This is in production.

Delivered approximately +360 VDI desktops for new users since March. 

 Resolved and documented  Azure CLI login issue by configuring custom CA bundle and appending Netskope certificate.

Completed request from the Identity Management team to publish the MySyscoAccess app, enabling access to the Identity Management tool for users outside of the VPN.

Successfully deleted additional VMs at GSC, reducing the count from 147 to just 7 remaining machines. Migration is on track to be completed by the end of June

Security Enhancements:

 Enabled global One-Time Passcode (OTP) MFA across all VDI environments.

Upgraded Forcepoint from version 24.11 to 25.04 | Upgraded Tanium from version 7.6.2.1259 to 7.6.4.2121 | Upgrade Netskope from version 117.1.3.2130 to 120.1.9.2256 | Installed windows11.0-kb5054980-x64 (.NET) windows11.0-kb5055528 | Installed windows10.0-kb5058379-x64

Upgrade the US based Horizon Connection Servers from 2309 to 2406 | Updated Chrome and Edge browsers to address critical vulnerabilities.
********************************************



Infrastructure Optimization and Cost Reduction:

 Resolved High CPU Usage in the VDI environment

 Renewed and implemented new external URL SSL  ws1-hzn.sysco.com on UAG and on F5

Decommissioned 42 Unused VMs and DVMs to optimize resource usage 

Major Accomplishments:

Successfully Completed VM Migration at the Global Support Center (GSC) 

 Deployed for production the policy to allow the use of time zone mapping to match users location. 

Migrated All Horizon Users to Sysco CLI v0.29 within App Volumes

Security Enhancements:

 Installed the monthly patches for Windows in the Horizon and AWS VDI
 
 
 
 *********************************************
 Optimización de Infraestructura y Reducción de Costos

Eliminación de ~140 VMs/DVMs de Horizon no utilizadas (limpieza del GSC casi completa, de 147 → 7).

Reducción de flotas de AppStream de ~44 → 9 (2 en Europa, 7 en EE. UU.).

Migración de cargas de trabajo VDI a nuevos clústeres de almacenamiento (Europa y EE. UU.).

Implementación de mapeo de zona horaria para usuarios globales.

Automatización de scripts de creación de VMs para agilizar migraciones del GSC.


Logros Principales

Entregadas +360 escritorios VDI nuevos desde marzo.

Publicación de MS Office en el grupo de escritorios Instant Clone de Windows 11 en la UE (producción).

Construcción y pruebas de grupos de escritorios Instant Clone de Windows 11 (UE).

Migración del equipo SEED a 39 nuevas VMs dedicadas de Horizon.

Migración de todos los usuarios de Horizon al Sysco CLI v0.29 (App Volumes).

Migración de VMs completada en el Global Support Center.

Resolución del problema de sincronización de Horizon y actualización a la versión 2406.

Soporte al equipo de Gestión de Identidad mediante la publicación de la aplicación MySyscoAccess.

Dashboards y alertas en Datadog completados para monitoreo de VDI.

Mejoras de Seguridad

Habilitación de MFA con OTP global en todos los entornos VDI.

Actualización de herramientas clave: Forcepoint (24.11 → 25.04), Tanium (7.6.2 → 7.6.4), Netskope (117 → 120), Servidores de Conexión de Horizon (2309 → 2406).

Aplicación de parches mensuales de Windows, .NET, Chrome, Edge y actualizaciones críticas de 7-Zip.

Optimización de Zscaler para tiempos de carga más rápidos en Horizon VDI de EE. UU.

Renovación e implementación de nuevo certificado SSL para acceso externo a Horizon.

SIgnuifucanlyy 




// 🎷 Good Morning Blues — Strudel arrangement (longer sections)
setcpm(120)

// ------------------ DRUMS ------------------
// Swing ride + snare backbeat (longer 2-bar phrase)
$: s("hh hh ~ hh hh hh ~ hh, ~ sd ~ ~ ~ sd ~ ~").bank("tr808")
.gain(0.2).dec(0.4).room(0.3).delay(0.02)

// Kick — alternating pattern with space (2-bar)
$: s("bd ~ ~ ~ bd ~ bd ~, ~ ~ bd ~ ~ bd ~ ~").bank("ajkpercusyn")
.gain(0.35).dec(0.5).room(0.4).slow(2)

// ------------------ BASS ------------------
// Walking bass, 4-bar cycle (I – IV – V blues movement)
$: note("g2 ~ a2 b2 | c3 ~ b2 a2 | d3 ~ c3 b2 | g2 ~ a2 g2")
.sound("gm_electric_bass_pick")
.gain(0.3).dec(2.5).room(0.8).legato(0.9).slow(2)

// ------------------ HARMONY ------------------
// Pad chords spanning 8 bars (bluesy I-IV-V progression)
$: note("<g3 b3 d4> ~ <c3 e3 g3> ~ | <d3 f3 a3> ~ <g3 b3 d4> ~")
.sound("gm_pad_1")
.gain(0.2).dec(4.0).room(0.7).legato(0.95).slow(4)

// ------------------ MELODY ------------------
// Trumpet motif (8-bar line with rests so it breathes)
$: note("~ g4 ~ ~ bb4 a4 ~ g4, f4 ~ ~ ~ d4 f4 g4 ~, ~ bb4 ~ a4 ~ ~ g4 ~")
.sound("gm_trumpet")
.gain(0.38).dec(1.2).room(0.35).delay(0.1).legato(0.7).pan(-0.15).slow(2)

// Alto Sax answer phrase (offset from trumpet, 8-bar)
$: note("~ ~ e4 g4 ~ bb4 a4 ~, g4 ~ f4 ~ ~ e4 g4 ~, a4 ~ ~ g4 f4 ~ ~ e4 ~")
.sound("gm_alto_sax")
.gain(0.34).dec(1.6).room(0.45).delay(0.18).legato(0.85).pan(0.15).slow(2)

// Good Morning Blues — Strudel patch (medium swing / shuffle)
// 3-voice arrangement: Vocal-pad, Trumpet, Alto Sax + rhythm + walking bass
setcpm(120)

// ------------------ DRUMS / GROOVE ------------------
// Kick layers for depth
$: s("~ ~ ~ bd ~ ~ ~ bd").bank("ajkpercusyn")
.gain(0.35).dec(0.4).room(0.25).delay(0.01)

$: s("~ ~ ~ bd ~ ~ ~ ~").bank("bossdr55")
.gain(0.28).dec(0.5).room(0.4).delay(0.5).legato(0.9)

// Snare on backbeat with a small room for snap
$: s("~ sd ~ ~ ~ sd ~ ~").bank("tr808")
.gain(0.30).dec(0.18).room(0.25).delay(0.01)

// Hi-hats (skippy / swung 8ths using slight delay variations)
$: s("~ hh ~ hh ~ hh ~ hh").bank("viscospacedrum")
.gain(0.015).dec(0.08).delay("<0.02 0.03 0.025>").room(0.8).legato(0.95)

// Shuffled percussion / ride to push the triplet feel
$: s("cp ~ ~ cp ~ ~ cp ~").bank("tr808")
.gain("<0.02 0.03>").delay("<0.18 0.35>").room(0.6).slow(1)

// ------------------ BASS / HARMONY ------------------
// Walking bass (2-bar repeating pattern — roots + passing notes)
$: note("g1 ~ a1 b1 ~ c2 ~ b1 a1 ~")
.sound("gm_electric_bass_pick")
.gain(0.25).dec(2.4).room(0.95).legato(0.92).slow(2)

// Skank / comp chords (offbeat stab to give the swing/blues comp)
$: note("~ g3 ~ g3 ~ g3 ~ g3").sound("gm_electric_guitar_jazz")
.gain(0.12).delay(0.2).room(0.45).legato(0.4).slow(1)

// Pad / vocal-like background (sustained chords that hold under solos)
$: note("<g3 b3 d4> ~ <g3 b3 d4> ~ ~ <c3 e3 g3> ~")
.sound("gm_pad_1")
.gain(0.18).dec(3.0).room(0.7).legato(0.95).slow(4)

// ------------------ LEADS (TRUMPET / ALTO SAX) ------------------
// Trumpet - short, punchy phrases, slightly left
$: note("~ g4 ~ <bb4 a4> g4 ~ f4 ~")
.sound("gm_trumpet")
.gain(0.40).dec(1.2).room(0.35).delay(0.12).legato(0.6).pan(-0.12)

// Alto Sax - smoother, answering phrases, slightly right
$: note("~ e4 ~ <g4 bb4> a4 ~ g4 ~")
.sound("gm_alto_sax")
.gain(0.32).dec(1.6).room(0.4).delay(0.18).legato(0.85).pan(0.18)

// Small harmonic fills to sit between vocal lines (muted/tucked)
$: note("~ ~ ~ <g4 a4> ~ ~ ~ <bb4 a4>").sound("sine")
.gain(0.10).dec(0.8).delay(0.28).room(0.55).legato(0.35).slow(2)

// ------------------ PERFORMANCE NOTES ------------------
// • This is an arrangement approximation (no lyrics included).
// • If your Strudel instance uses different GM/sample names, replace
//   "gm_trumpet", "gm_alto_sax", "gm_pad_1" etc. with your available samples.
// • To emphasize the shuffle feel, increase the hi-hat/ride delay differences
//   (see the <...> arrays used in .delay on hats/ride).
// • To create a head → trumpet solo → alto solo → duet form you can:
//     - manually mute/unmute the trumpet or alto in the REPL UI, OR
//     - replace the trumpet/alto patterns with longer/resting patterns using
//       more `~`s and `.slow()` to emulate section lengths (I kept all parts live
//       so you can audition and then shape structure in the REPL).
//
// Want it further polished? I can:
// 1) map a real trumpet & sax WAV sample set (give me names/URLs) and swap the .sound(...) calls, OR
// 2) convert a short melody phrase you provide into a trumpet `note(...)` line with exact pitches and rhythm.


\\ms248hznmgmt01


Is the user enamble -> FInd user in an AD group if users are enabled
tche2579

Root of DVM profile 

lsri7480

247-W11-DVM067
247-WIN11-DVM30

AWS AppStream Optimization – Cost Reduction Journey

2022 → July 2025 = 35 months of data.
Starting point (Sept 2022): $11,481.32
Latest point (July 2025): $6,950.00
Total reduction: ~$4,531.32/month (≈ 39.5% cost reduction)

If we take averages:
Average 2022: ≈ $10,924
Average 2023: ≈ $9,022
Average 2024: ≈ $7,264
Average 2025: ≈ $6,900


~40% reduction in monthly spend (Sep 2022 → Jul 2025).
Costs dropped from $11.5K → $6.9K/month.
Annualized savings ≈ $54K/year.
More efficient service delivery with no impact on user experience.







********************************************
Run-time error '1004': Method 'SaveAs' of object '_Workbook' failed

This usually happens in Excel VBA when the Workbook.SaveAs method cannot complete. 

Invalid File Path or Name

If your VBA macro tries to save the file with characters not allowed in Windows (\ / : * ? " < > |), the save will fail.
Check the filename and path.

File Already Open or Locked
If the target file is open (maybe by you or another process), SaveAs will fail. 
Close the existing file before running the macro.

Insufficient Permissions
If saving to a protected location (e.g., C:\, Program Files, or a network folder with restrictions).
Save to a user folder like Documents or Desktop.

Wrong File Format
If you're saving with .xlsx but using a macro-enabled workbook with VBA, Excel requires .xlsm.
Match extension with format:




File Already Exists (Overwrite Issue)
Some versions of Excel prompt to overwrite and VBA doesn’t handle it.
Delete or rename the old file before saving.

************************

Hello Team,

We have encountered an issue "
INC000006136631" where Excel macros are failing with a “Run-time error 1004: Method 'SaveAs' of object '_Workbook' failed” when attempting to save files to OneDrive.

Since the VDI team does not handle application-level issues and no action plan exists on our side for Excel macro failures, I am cancelling the incodent and requesting this matter for your support and investigation.

Please reach out to the end user for resolving this issue as it out of our scipe of support.
