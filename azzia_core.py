import torch
import random
import numpy as np
from PIL import Image
import io
import base64
import requests
import json
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256
import folder_paths
import comfy.sd

def encrypt_aes_v2(data, password):
    """
    Cifra usando AES-256-CBC con PBKDF2-SHA256 para derivación de clave.
    Formato compatible con la implementación V2 del frontend (CryptoJS).
    """
    SALT_SIZE = 16
    KEY_SIZE = 32
    IV_SIZE = 16
    ITERATIONS = 600000
    VERSION_PREFIX = "AZZIA_V2$"

    salt = get_random_bytes(SALT_SIZE)
    iv = get_random_bytes(IV_SIZE)
    key = PBKDF2(password, salt, dkLen=KEY_SIZE, count=ITERATIONS, hmac_hash_module=SHA256)
    
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_data = pad(data.encode('utf-8'), AES.block_size)
    ciphertext = cipher.encrypt(padded_data)

    payload = {
        "s": base64.b64encode(salt).decode('utf-8'),
        "iv": base64.b64encode(iv).decode('utf-8'),
        "ct": base64.b64encode(ciphertext).decode('utf-8')
    }
    
    return VERSION_PREFIX + json.dumps(payload)


class TextPassthrough:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "passthrough"
    CATEGORY = "Azzia_Nodes"

    def passthrough(self, text):
        return (text,)


class SeedCapture:
    last_seed = 0
    execution_id = 0
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("seed", "seed_copy")
    FUNCTION = "capture"
    CATEGORY = "Azzia_Nodes"

    def capture(self, seed):
        # Reset LoRA list al inicio de un nuevo workflow
        # (SeedCapture siempre se ejecuta antes que PostImageToAPI)
        # LoraLoaderCapture.reset_loras() # MOVIDO a PostImageToAPI para evitar race conditions
        
        SeedCapture.last_seed = seed
        SeedCapture.execution_id += 1
        print(f"🎲 Seed capturado: {seed}")
        return (seed, seed)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")


class CheckpointCapture:
    """
    Wrapper del CheckpointLoaderSimple que captura el nombre del modelo automáticamente.
    Para arquitecturas SD1.5 / SDXL (checkpoint único con MODEL+CLIP+VAE).
    """
    last_model_name = ""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "Azzia_Nodes"

    def load_checkpoint(self, ckpt_name):
        # Capturar el nombre del modelo
        CheckpointCapture.last_model_name = ckpt_name
        print(f"🤖 Modelo capturado: {ckpt_name}")
        
        # Cargar el checkpoint usando la función nativa de ComfyUI
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, 
            output_vae=True, 
            output_clip=True, 
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )
        
        return out[:3]


class UNETCapture:
    """
    Wrapper del UNETLoader para arquitecturas Flux / SD3.
    Captura el nombre del modelo UNET automáticamente para PostImageToAPI.
    REEMPLAZA al UNETLoader original en workflows Flux — usa este en su lugar.
    """
    last_unet_name = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2",
                                   "fp8_e4m3fn_fast", "bf16", "fp16", "fp32"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "Azzia_Nodes"

    def load_unet(self, unet_name, weight_dtype):
        # Capturar el nombre de la UNET Flux
        UNETCapture.last_unet_name = unet_name
        print(f"🧠 UNET Flux capturada: {unet_name}")

        # Construir model_options según el dtype elegido
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16
        elif weight_dtype == "fp16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "fp32":
            model_options["dtype"] = torch.float32

        # Cargar usando la función nativa de ComfyUI para modelos Flux/SD3
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)


class LoraLoaderCapture:
    """
    Wrapper del LoraLoader que captura automáticamente los parámetros
    REEMPLAZA al LoraLoader original - usa este en lugar del LoraLoader de ComfyUI
    """
    lora_list = []
    execution_counter = 0
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "Azzia_Nodes"

    @classmethod
    def IS_CHANGED(s, lora_name="", strength_model=1.0, strength_clip=1.0, **kwargs):
        # Retorna hash estable basado en los parámetros del LoRA.
        # Solo se re-ejecuta si cambia el nombre o los strengths.
        # Esto evita re-encoding innecesario del CLIP cuando el prompt es el mismo.
        import hashlib
        key = f"{lora_name}:{strength_model}:{strength_clip}"
        return hashlib.md5(key.encode()).hexdigest()

    @classmethod
    def reset_loras(cls):
        cls.lora_list = []

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        # Capturar automáticamente los parámetros
        lora_info = {
            "name": lora_name,
            "strength_model": strength_model,
            "strength_clip": strength_clip
        }
        
        # Verificar si ya existe con los mismos valores
        existing = next((l for l in LoraLoaderCapture.lora_list if l["name"] == lora_name), None)
        
        if existing:
            # Actualizar los valores si cambiaron
            if existing["strength_model"] != strength_model or existing["strength_clip"] != strength_clip:
                existing["strength_model"] = strength_model
                existing["strength_clip"] = strength_clip
                print(f"✨ LoRA actualizado: {lora_name} (model: {strength_model}, clip: {strength_clip})")
        else:
            # Agregar nuevo LoRA
            LoraLoaderCapture.lora_list.append(lora_info)
            print(f"✨ LoRA capturado automáticamente: {lora_name} (model: {strength_model}, clip: {strength_clip})")
        
        # Cargar el LoRA usando la función nativa de ComfyUI
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
        
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        
        return (model_lora, clip_lora)


class PostImageToAPI:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "password": ("STRING", {"default": "", "placeholder": "🔑 Contraseña de cifrado"}),
                "endpoint_url": ("STRING", {"default": "https://corelink.onrender.com/azzia/prompt"}),
            },
            "optional": {
                "positive_prompt": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_image"
    OUTPUT_NODE = True
    CATEGORY = "Azzia_Nodes"

    def send_image(self, images, password, endpoint_url, positive_prompt="", negative_prompt="", seed=0):
        if not password:
            print("❌ ERROR: No se proporcionó contraseña para el cifrado AES.")
            return {}

        # Capturar datos automáticamente
        if seed == 0 and hasattr(SeedCapture, 'last_seed'):
            seed = SeedCapture.last_seed

        # Soporta SD1.5/SDXL (CheckpointCapture) y Flux/SD3 (UNETCapture)
        model_name = (getattr(CheckpointCapture, 'last_model_name', "") or
                      getattr(UNETCapture, 'last_unet_name', ""))
        loras = getattr(LoraLoaderCapture, 'lora_list', []).copy()

        # Construir metadata
        metadata_parts = []
        
        if model_name:
            metadata_parts.append(f"Model: {model_name}")
        
        if loras:
            lora_names = []
            for lora in loras:
                lora_str = f"{lora['name']}"
                if lora['strength_model'] != 1.0 or lora['strength_clip'] != 1.0:
                    lora_str += f" ({lora['strength_model']}/{lora['strength_clip']})"
                lora_names.append(lora_str)
            metadata_parts.append(f"LoRAs: {', '.join(lora_names)}")
        
        if seed:
            metadata_parts.append(f"Seed: {seed}")
        
        metadata_text = " | ".join(metadata_parts) if metadata_parts else ""
        
        if metadata_text:
            final_prompt = f"{positive_prompt} ({metadata_text})"
        else:
            final_prompt = positive_prompt

        neg_prompt = negative_prompt if negative_prompt else ""

        for image in images:
            # 1. Procesar Imagen
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            base64_image = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
            
            # 2. CIFRAR DATOS
            print("🔐 Cifrando datos con AES V2 (Azzia Secure Standard)...")
            # Los prints de prompts han sido eliminados por privacidad
            
            try:
                enc_image = encrypt_aes_v2(base64_image, password)
                enc_prompt = encrypt_aes_v2(final_prompt, password)
                enc_neg_prompt = encrypt_aes_v2(neg_prompt, password)
                
                payload = {
                    "prompt": enc_prompt,
                    "negativePrompt": enc_neg_prompt,
                    "imagenEncriptada": enc_image
                }
                
                response = requests.post(endpoint_url, json=payload, timeout=60)
                if response.status_code in [200, 201]:
                    print(f"🚀 ENVIADOOoOoOo Y CIFRADO: Status {response.status_code}")
                else:
                    print(f"⚠️ Error {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error en el proceso de envío/cifrado: {e}")

        # Limpiar lista de LoRAs después de enviar
        LoraLoaderCapture.reset_loras()
        return {}


class AtomicPromptGenerator:
    """
    Genera prompts variados combinando slots atómicos independientes.
    Usa anti-repetición por cooldown y compatibilidad semántica por tags.
    """

    # --- Base de datos interna (fallback) ---
    _DB = {
        "verb": [
            {"token": "playing",       "tags": ["dynamic", "casual"]},
            {"token": "running",       "tags": ["dynamic", "energetic"]},
            {"token": "sitting",       "tags": ["calm", "casual"]},
            {"token": "stretching",    "tags": ["dynamic", "sensual"]},
            {"token": "gazing",        "tags": ["calm", "dreamy"]},
            {"token": "wading",        "tags": ["calm", "outdoor"]},
            {"token": "resting",       "tags": ["calm", "casual"]},
            {"token": "posing",        "tags": ["sensual", "editorial"]},
            {"token": "leaning",       "tags": ["casual", "editorial"]},
            {"token": "smiling",       "tags": ["casual", "warm"]},
            {"token": "laughing",      "tags": ["dynamic", "warm"]},
            {"token": "lying down",    "tags": ["calm", "sensual"]},
        ],
        "adverb": [
            {"token": "playfully",     "tags": ["dynamic", "casual"]},
            {"token": "gracefully",    "tags": ["calm", "editorial"]},
            {"token": "lazily",        "tags": ["calm", "casual"]},
            {"token": "intensely",     "tags": ["dynamic", "dramatic"]},
            {"token": "dreamily",      "tags": ["calm", "dreamy"]},
            {"token": "confidently",   "tags": ["editorial", "sensual"]},
            {"token": "quietly",       "tags": ["calm", "dreamy"]},
            {"token": "sensually",     "tags": ["sensual", "editorial"]},
        ],
        "location": [
            {"token": "on the beach",          "tags": ["outdoor", "warm"]},
            {"token": "in the waves",           "tags": ["outdoor", "warm"]},
            {"token": "near the rocks",         "tags": ["outdoor", "dramatic"]},
            {"token": "at the shoreline",       "tags": ["outdoor", "warm"]},
            {"token": "on the pier",            "tags": ["outdoor", "casual"]},
            {"token": "by a swimming pool",     "tags": ["outdoor", "warm"]},
            {"token": "in a tropical garden",   "tags": ["outdoor", "warm"]},
            {"token": "on a rooftop terrace",   "tags": ["outdoor", "urban"]},
            {"token": "in an alley",            "tags": ["urban", "dramatic"]},
            {"token": "in a white studio",      "tags": ["indoor", "editorial"]},
            {"token": "on a bed",               "tags": ["indoor", "sensual"]},
            {"token": "in soft window light",   "tags": ["indoor", "calm"]},
        ],
        "angle": [
            {"token": "wide shot",              "tags": ["outdoor", "dynamic"]},
            {"token": "close-up portrait",      "tags": ["editorial", "calm"]},
            {"token": "low angle shot",         "tags": ["dramatic", "dynamic"]},
            {"token": "overhead shot",          "tags": ["outdoor", "casual"]},
            {"token": "side profile",           "tags": ["editorial", "sensual"]},
            {"token": "dutch angle",            "tags": ["dramatic", "urban"]},
            {"token": "medium shot",            "tags": ["casual", "editorial"]},
            {"token": "three-quarter view",     "tags": ["editorial", "sensual"]},
        ],
        "lighting": [
            {"token": "golden hour light",      "tags": ["warm", "outdoor"]},
            {"token": "soft morning light",     "tags": ["calm", "outdoor"]},
            {"token": "dramatic sunset",        "tags": ["dramatic", "outdoor"]},
            {"token": "overcast diffused light","tags": ["calm", "outdoor"]},
            {"token": "neon reflections",       "tags": ["urban", "dramatic"]},
            {"token": "studio softbox light",   "tags": ["indoor", "editorial"]},
            {"token": "candlelight",            "tags": ["indoor", "warm"]},
            {"token": "hard sunlight",          "tags": ["outdoor", "dynamic"]},
            {"token": "rimlight",               "tags": ["dramatic", "editorial"]},
        ],
        "mood": [
            {"token": "cinematic",      "tags": ["dramatic", "editorial"]},
            {"token": "editorial",      "tags": ["editorial", "calm"]},
            {"token": "candid",         "tags": ["casual", "dynamic"]},
            {"token": "serene",         "tags": ["calm", "dreamy"]},
            {"token": "vibrant",        "tags": ["dynamic", "warm"]},
            {"token": "ethereal",       "tags": ["dreamy", "calm"]},
            {"token": "provocative",    "tags": ["sensual", "editorial"]},
            {"token": "dreamy",         "tags": ["dreamy", "calm"]},
        ],
    }

    # Historial de cooldown por slot
    _history: dict = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "subject":  ("STRING", {"default": "a girl",                                              "multiline": False}),
                "traits":   ("STRING", {"default": "red hair, blue eyes, freckles",                       "multiline": False}),
                "outfit":   ("STRING", {"default": "red swimsuit",                                        "multiline": False}),
                "quality":  ("STRING", {"default": "4k, raw photo, sharp focus, realistic skin texture, f/1.8", "multiline": False}),
                "cooldown": ("INT",    {"default": 3, "min": 1, "max": 10}),
                "seed":     ("INT",    {"default": -1, "min": -1, "max": 0xffffffffffffffff,
                                        "tooltip": "-1 = completamente aleatorio cada vez"}),
            },
            "optional": {
                "db_file": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "C:/ruta/a/tu/db.json  (dejar vacío = usar DB interna)"
                }),
                "db_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Pega aquí tu JSON de slots (opcional, se ignora si db_file está activo)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate"
    CATEGORY = "Azzia_Nodes"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    @classmethod
    def _normalize_slot(cls, items: list) -> list:
        out = []
        for item in items:
            if isinstance(item, str):
                out.append({"token": item, "tags": []})
            elif isinstance(item, dict) and "token" in item:
                out.append({"token": item["token"], "tags": item.get("tags", [])})
        return out

    @classmethod
    def _load_db(cls, db_file: str = "", db_json: str = "") -> tuple:
        raw = None
        if db_file and db_file.strip():
            try:
                path = db_file.strip()
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                print(f"⚛️  DB cargada desde archivo: {path}")
            except Exception as e:
                print(f"⚠️  Error leyendo '{db_file}': {e}")

        if raw is None and db_json and db_json.strip():
            try:
                raw = json.loads(db_json.strip())
                print("⚛️  DB cargada desde JSON inline")
            except Exception as e:
                print(f"⚠️  JSON inline inválido: {e}")

        if raw is None:
            return cls._DB, {}

        slots_raw      = raw.get("slots", raw)
        char_overrides = raw.get("character", {})

        merged = {}
        for slot_name, default_pool in cls._DB.items():
            if slot_name in slots_raw:
                merged[slot_name] = cls._normalize_slot(slots_raw[slot_name])
            else:
                merged[slot_name] = default_pool

        return merged, char_overrides

    @classmethod
    def _pick_fresh(cls, db: dict, slot_name: str, cooldown: int) -> dict:
        pool = db[slot_name]
        used = cls._history.get(slot_name, [])
        available = [t for t in pool if t["token"] not in used]
        if not available:
            cls._history[slot_name] = []
            available = pool
        chosen = random.choice(available)
        history = cls._history.get(slot_name, [])
        history.append(chosen["token"])
        if len(history) > cooldown:
            history = history[-cooldown:]
        cls._history[slot_name] = history
        return chosen

    @classmethod
    def _pick_compatible(cls, db: dict, anchor_slot: str, target_slot: str, cooldown: int) -> dict:
        anchor_history = cls._history.get(anchor_slot, [])
        if not anchor_history:
            return cls._pick_fresh(db, target_slot, cooldown)
        anchor_token_name = anchor_history[-1]
        anchor_entry = next((t for t in db[anchor_slot] if t["token"] == anchor_token_name), None)
        if not anchor_entry or not anchor_entry.get("tags"):
            return cls._pick_fresh(db, target_slot, cooldown)
        anchor_tags = set(anchor_entry["tags"])
        pool        = db[target_slot]
        used        = cls._history.get(target_slot, [])
        compatible = [t for t in pool if t["token"] not in used and bool(set(t.get("tags", [])) & anchor_tags)]
        if not compatible:
            return cls._pick_fresh(db, target_slot, cooldown)
        chosen = random.choice(compatible)
        history = cls._history.get(target_slot, [])
        history.append(chosen["token"])
        if len(history) > cooldown:
            history = history[-cooldown:]
        cls._history[target_slot] = history
        return chosen

    def generate(self, subject, traits, outfit, quality, cooldown, seed, db_file="", db_json=""):
        if seed != -1:
            random.seed(seed)
        db, char = self._load_db(db_file, db_json)
        subject = char.get("subject", subject)
        traits  = char.get("traits",  traits)
        outfit  = char.get("outfit",  outfit)
        quality = char.get("quality", quality)
        verb     = self._pick_fresh(db, "verb", cooldown)
        adverb   = self._pick_compatible(db, "verb",     "adverb",   cooldown)
        lighting = self._pick_fresh(db, "lighting", cooldown)
        location = self._pick_compatible(db, "lighting", "location", cooldown)
        angle    = self._pick_compatible(db, "lighting", "angle",    cooldown)
        mood     = self._pick_compatible(db, "lighting", "mood",     cooldown)
        prompt = (
            f"{subject} {verb['token']} {adverb['token']} {location['token']}, "
            f"{angle['token']}, {lighting['token']}, {mood['token']} mood, "
            f"wearing {outfit}, {traits}, {quality}"
        )
        print(f"🎨 Atomic Prompt generado: {prompt}")
        return (prompt,)


class FullPromptInjector:
    """
    Inyecta prompts completos de forma aleatoria desde una lista JSON.
    El usuario escribe los prompts completos — no hay generación dinámica.
    Usa anti-repetición por cooldown para no repetir el mismo prompt seguido.
    """

    # Historial de cooldown
    _history: list = []

    # Último prompt generado (para IS_CHANGED)
    _last_prompt: str = ""

    # DB interna de ejemplo (fallback si no se provee archivo)
    _DEFAULT_PROMPTS = [
        "a beautiful woman sitting on the beach, golden hour light, wide shot, cinematic mood, 4k, raw photo, sharp focus",
        "a beautiful woman posing in a white studio, studio softbox light, editorial mood, close-up portrait, 4k",
        "a beautiful woman gazing dreamily by a swimming pool, soft morning light, serene mood, medium shot, 4k",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed":     ("INT", {
                    "default": -1, "min": -1, "max": 0xffffffffffffffff,
                    "tooltip": "-1 = completamente aleatorio cada vez"
                }),
                "cooldown": ("INT", {"default": 3, "min": 1, "max": 50,
                    "tooltip": "Cuántos prompts se recuerdan para evitar repetición"
                }),
            },
            "optional": {
                "db_file": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "C:/ruta/a/tu/prompts.json  (dejar vacío = usar DB interna)"
                }),
                "db_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": 'Pega aquí tu JSON: {"prompts": ["prompt 1", "prompt 2", ...]}'
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "inject"
    CATEGORY = "Azzia_Nodes"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    @classmethod
    def _load_prompts(cls, db_file: str = "", db_json: str = "") -> list:
        """Carga la lista de prompts completos desde archivo, JSON inline, o DB interna."""
        if db_file and db_file.strip():
            try:
                with open(db_file.strip(), "r", encoding="utf-8") as f:
                    data = json.load(f)
                prompts = data.get("prompts", data) if isinstance(data, dict) else data
                if isinstance(prompts, list) and prompts:
                    print(f"📝 FullPromptInjector: {len(prompts)} prompts cargados desde {db_file.strip()}")
                    return [str(p) for p in prompts if str(p).strip()]
            except Exception as e:
                print(f"⚠️  FullPromptInjector: Error leyendo '{db_file}': {e}")

        if db_json and db_json.strip():
            try:
                data = json.loads(db_json.strip())
                prompts = data.get("prompts", data) if isinstance(data, dict) else data
                if isinstance(prompts, list) and prompts:
                    print(f"📝 FullPromptInjector: {len(prompts)} prompts cargados desde JSON inline")
                    return [str(p) for p in prompts if str(p).strip()]
            except Exception as e:
                print(f"⚠️  FullPromptInjector: JSON inline inválido: {e}")

        print("📝 FullPromptInjector: Usando DB interna de ejemplo")
        return cls._DEFAULT_PROMPTS

    def inject(self, seed, cooldown, db_file="", db_json=""):
        if seed != -1:
            random.seed(seed)

        prompts = self._load_prompts(db_file, db_json)

        # Anti-repetición por cooldown
        used = self.__class__._history
        available = [p for p in prompts if p not in used]
        if not available:
            self.__class__._history = []
            available = prompts

        chosen = random.choice(available)

        # Actualizar historial
        history = self.__class__._history
        history.append(chosen)
        if len(history) > cooldown:
            history = history[-cooldown:]
        self.__class__._history = history
        self.__class__._last_prompt = chosen

        print(f"💬 FullPromptInjector → {chosen[:80]}{'...' if len(chosen) > 80 else ''}")
        return (chosen,)

NODE_CLASS_MAPPINGS = {
    "PostImageToAPI": PostImageToAPI,
    "TextPassthrough": TextPassthrough,
    "SeedCapture": SeedCapture,
    "CheckpointCapture": CheckpointCapture,
    "UNETCapture": UNETCapture,
    "LoraLoaderCapture": LoraLoaderCapture,
    "AtomicPromptGenerator": AtomicPromptGenerator,
    "FullPromptInjector": FullPromptInjector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PostImageToAPI": "🚀 Send Encrypted Image to Azzia",
    "TextPassthrough": "📝 Text Passthrough",
    "SeedCapture": "🎲 Seed Capture",
    "CheckpointCapture": "🤖 Checkpoint Loader (Auto-Capture)",
    "UNETCapture": "🧠 UNET Loader Flux (Auto-Capture)",
    "LoraLoaderCapture": "✨ LoRA Loader (Auto-Capture)",
    "AtomicPromptGenerator": "⚛️ Atomic Prompt Generator",
    "FullPromptInjector": "💬 Full Prompt Injector",
}