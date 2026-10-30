# Final Course Design and Validation

## Interaction Grammar

The HTML and PowerPoint outputs use the same synchronized model:

```text
exact source  <->  object / IR / hardware visualization  <->  state + math + evidence
                                  |
                           shared build state
```

Each HTML scene has a contract state plus four progressive builds. Each PowerPoint build slide contains three On Click groups: newly introduced source lines, the corresponding system transition, and the state/math explanation.

## Inventory

- 586 slides across six decks
- 568 animated slides
- 1704 click groups
- 142 HTML source scenes
- 71 source units
- 41 source files

## Evidence Rules

- `SOURCE`: exact source proves the displayed contract or transformation.
- `BOUNDARY`: a native compiler, serialized artifact, vendor API, or hardware implementation is below the displayed source.
- `COMPILE-ONLY`: target lowering is inspected without claiming target execution.
- `GPU-LAB`: timing, traffic, overlap, occupancy, and throughput require a future target GPU measurement.
- Qualcomm is not presented as a supported MAX GPU backend. The source atlas marks the Hexagon/HTP discussion as a boundary.

## Validation

Run `npm run check` after generation. It verifies source/anchor resolution, slide counts, notes, transitions, animation timing XML, HTML links, and generated asset syntax. Browser screenshots and PowerPoint rendering checks are performed separately before delivery.
