from pathlib import Path
import geopandas as gpd
import zipfile
import shutil

BASE_DIR = Path(__file__).resolve().parent

# 1. Localiza dinamicamente o arquivo .zip (em processos_filtrados ou shapes)
INPUT_DIR = BASE_DIR / "processos"
zip_files = list(INPUT_DIR.glob("*.zip"))

if not zip_files:
    raise FileNotFoundError("Nenhum arquivo .zip foi encontrado em 'processos' ou 'shapes'.")

ZIP_FILE = zip_files[0]
print(f"Arquivo ZIP identificado: {ZIP_FILE.name}")

TEMP_DIR = BASE_DIR / "temp_extracted"
OUTPUT_DIR = BASE_DIR / "shapes"

# 2. Extrai os arquivos do ZIP para uma pasta temporária
print("Extraindo arquivos...")
with zipfile.ZipFile(ZIP_FILE, 'r') as z:
    z.extractall(TEMP_DIR)

# 3. Procura o arquivo .shp extraído
shp_files = list(TEMP_DIR.rglob("*.shp"))
if not shp_files:
    raise FileNotFoundError("Nenhum arquivo .shp foi encontrado dentro do ZIP.")

SHP_PATH = shp_files[0]

# 4. Remove o arquivo .prj com EPSG:404000 corrompido para não quebrar o pyproj
prj_path = SHP_PATH.with_suffix(".prj")
if prj_path.exists():
    prj_path.unlink()

print("Lendo Shapefile...")
gdf = gpd.read_file(SHP_PATH)

# Atribui o CRS correto da ANM (SIRGAS 2000)
gdf.crs = "EPSG:4674"

print(f"Exportando {len(gdf)} processos para pastas individuais...")

print("=-"*70)
entrada_usuario = input("Digite o número inicial para a sequência dos shapes (ex: 340 gerará 340, 341, 342...). Para usar o padrão, apenas pressione Enter: ")
print("=-"*70)
number = int(entrada_usuario) if entrada_usuario.strip() else None

# 5. Agrupa e salva cada processo em sua subpasta
for proc_val, group in gdf.groupby("processo"):
    # Trata a barra (ex: 855262/1995 -> 855262_1995 ou 1/2026 -> 1_2026)
    proc_clean = str(proc_val).replace("/", "_").strip()
    
    if number is not None:
        try:
            prefixo, ano = proc_clean.split("_", 1)
            
            novo_prefixo = int(prefixo) + number - 1
            
            proc_final = f"{novo_prefixo}_{ano}"
        except ValueError:
            proc_final = proc_clean
    else:
        proc_final = proc_clean

    folder_name = f"processo_{proc_final}"
    id_dir = OUTPUT_DIR / folder_name
    id_dir.mkdir(parents=True, exist_ok=True)

    out_file = id_dir / f"{folder_name}.shp"
    group.to_file(out_file, encoding="utf-8")

    shutil.make_archive(OUTPUT_DIR / folder_name, 'zip', id_dir)
    shutil.rmtree(id_dir)  # Remove a pasta temporária após criar o ZIP
# 6. Limpa a pasta temporária de extração
shutil.rmtree(TEMP_DIR)

print(f"Concluído! Todos os {len(gdf)} arquivos foram salvos em: {OUTPUT_DIR}")