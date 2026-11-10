# Hardware: to be defined

Nothing in this repo depends on the hardware yet. Fill this in when the specs
are available, then implement `sdl/backends/real.py`.

| Item | Needed for | Status |
|---|---|---|
| Robot arm model and control API (SDK, ROS, serial?) | `RealBackend.add/stir/discard` | TBD |
| How liquid is dispensed (pipette tool, syringe pump, pre-filled vials?) | `RealBackend.add`, `limits.min/max_dispense_ml`, `faults.volume_cv` | TBD |
| Magnetic stirrer control (manual, relay, serial?) | `RealBackend.stir`, `limits.max_stir_s` | TBD |
| Camera model, resolution, exposure and white-balance control | `RealBackend.image`, `camera.*` priors | TBD |
| Vial type, inner diameter (optical path), capacity | `optics.path_cm`, `limits.vial_capacity_ml` | TBD |
| Position of the white reference patch, vial and black stripe in the frame | `sdl/vision/layout.py` | TBD |
| Lighting (type, diffuser, enclosure) | `camera.session_*` priors | TBD |

## Assumptions made in the meantime

- The camera views the vial from the side against a white backdrop, so the
  optical path is the vial's inner diameter (taken as 1 cm).
- A white reference patch is always in frame (white balance).
- A black stripe sits on the backdrop behind the vial (turbidity readout). This
  is an addition to the original PoC setup and costs a strip of black tape.
- Camera auto-exposure and auto-white-balance are switched off.
