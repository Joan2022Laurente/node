# azzia-nodes

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — parte del ecosistema **Azzia / Corelink**.

## Instalación

### Opción A — ComfyUI Manager (recomendado)
1. Abre ComfyUI Manager → *Install via Git URL*
2. Pega la URL de este repositorio
3. Reinicia ComfyUI

### Opción B — Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/TU_USUARIO/azzia-nodes.git
cd azzia-nodes
pip install -r requirements.txt
```
Reinicia ComfyUI.

---

## Nodos incluidos

| Nodo | Categoría | Descripción |
|------|-----------|-------------|
| 🚀 **Send Encrypted Image to Azzia** | `Azzia_Nodes` | Envía la imagen generada + prompts + metadata a la API Azzia cifrado con AES-256-CBC |
| 🧠 **UNET Loader Flux (Auto-Capture)** | `Azzia_Nodes` | Wrapper de `UNETLoader` para arquitecturas Flux/SD3 — captura el nombre del modelo automáticamente |
| 🤖 **Checkpoint Loader (Auto-Capture)** | `Azzia_Nodes` | Wrapper de `CheckpointLoaderSimple` para SD1.5/SDXL — captura el modelo automáticamente |
| ✨ **LoRA Loader (Auto-Capture)** | `Azzia_Nodes` | Wrapper de `LoraLoader` — captura nombre y strengths del LoRA automáticamente |
| 🎲 **Seed Capture** | `Azzia_Nodes` | Captura el seed y lo pasa al sampler + a la API |
| 📝 **Text Passthrough** | `Azzia_Nodes` | Pasa un texto como STRING — permite capturar prompts para enviarlos a la API |
| ⚛️ **Atomic Prompt Generator** | `Azzia_Nodes` | Genera prompts variados combinando slots atómicos con anti-repetición por cooldown y compatibilidad semántica |

---

## Workflow de ejemplo

El archivo `ZIMAGET2I_lora.json` en la raíz del proyecto es un workflow de referencia para **Flux** (Text-to-Image con LoRA) que incluye todos los nodos de captura conectados a la API Azzia.

Cómo cargarlo: ComfyUI → *Load* → selecciona `ZIMAGET2I_lora.json`

---

## Configuración del nodo `PostImageToAPI`

| Campo | Descripción |
|-------|-------------|
| `password` | Contraseña para cifrado AES-256-CBC (debe coincidir con la configurada en tu backend Corelink) |
| `endpoint_url` | URL de tu API Azzia (por defecto: `https://corelink.onrender.com/azzia/prompt`) |

> ⚠️ **Nunca subas tu contraseña al repositorio.** Ingrésala directamente en el nodo dentro de ComfyUI.

---

## Requisitos

- ComfyUI (versión con soporte Flux — `UNETLoader` disponible)
- Python 3.10+
- Ver `requirements.txt` para dependencias Python

---

## Arquitecturas soportadas

| Arquitectura | Loader a usar |
|---|---|
| SD 1.5 / SDXL | `Checkpoint Loader (Auto-Capture)` |
| Flux / SD3 | `UNET Loader Flux (Auto-Capture)` |
