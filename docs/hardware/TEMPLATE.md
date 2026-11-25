# <Device name>

> Copy to `docs/hardware/<device>.md`. Fill every field, or write `unknown: how to find out`.
> Purpose: facts for **protocol step 3** (valid cells) and **step 5** (setup). Datasheet TOPS don't rank devices.

| Field | Value | Source / how verified |
|---|---|---|
| Exact model / revision | | label, `lsusb`, `cat /proc/device-tree/model`… |
| OS / firmware we use (why) | | |
| Compute (CPU / accelerator, RAM) | | |
| Formats & precisions (FP32/FP16/INT8) | | |
| Toolchain + version | | |
| Host requirements to compile | | |
| Link to host (USB 2/3, PCIe gen/lanes) + check command | | `lsusb -t`, `lspci -vv` |
| Built-in sensors (power/temp/clocks/throttling) + exact commands | | |
| Power: datasheet + PSU we use | | |
| Known pitfalls (CPU fallback, throttling, cooling, drivers) | | |

## Device-specific extras
(see the issue for this device)

## Sources
- 
