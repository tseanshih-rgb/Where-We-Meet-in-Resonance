# Where We Meet in Resonance

A multi-person interactive sound installation that makes unspoken social distance audible. A camera tracks visitors in real time, and their proximity and movement shape the pitch, timbre and spatial placement of a generative soundscape. Visitors become the instrument: as they gather, their sounds converge toward harmony and resonance.

Selected for *Dream ex Machina*, exhibited at Tekniska museet in collaboration with NAVET.

**Video & photos:** https://youtu.be/GhTWPjP4Ivo

## How it works

- **Tracking:** a Python script, adapted from an existing YOLOv8 model, detects visitors and sends their positions, speed and group measures (dispersion, nearest distance, head-count, encounters) to Max/MSP over OSC/UDP.
- **Sound:** each visitor is one voice in Max/MSP. As the group gathers, voices glide toward a shared chord and the timbre clears; as people drift apart, the sound becomes dissonant and diffuse. Movement speed shapes loudness and brightness.
- **Spatial audio:** each person's position is converted to an azimuth and panned across a four-speaker array.
- **Sound design:** the Max output is routed into Ableton Live for sound design and returned to Max for the final four-channel mix.
