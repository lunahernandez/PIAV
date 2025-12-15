import os
import itertools
import re

def extraer_id_persona(nombre_archivo):
    """
    Extrae el ID numérico del nombre del archivo.
    Ejemplos:
      'crd_0815f_01.png' -> '0815'
      'crd_0815s_01.png' -> '0815'
      'crd_0813f_01.png' -> '0813'
    """
    # Busca un patrón de 4 dígitos en el nombre
    match = re.search(r'(\d{4})', nombre_archivo)
    if match:
        return match.group(1)
    return None

def generar_pairs_txt_lfw_from_users(root_dir, users, output_txt, ext=".png"):
    imgs_por_id = {}  # Agrupar por ID numérico

    for user in users:
        folder_abs = os.path.join(root_dir, user)
        if not os.path.isdir(folder_abs):
            continue

        # Listar todas las imágenes en la carpeta del usuario
        archivos = [f for f in os.listdir(folder_abs) if f.lower().endswith(ext)]
        
        for archivo in archivos:
            # Extraer el ID de la persona desde el nombre del archivo
            person_id = extraer_id_persona(archivo)
            if person_id is None:
                print(f"Warning: No se pudo extraer ID de {archivo}")
                continue
            
            # Ruta relativa del archivo
            img_rel = os.path.join(user, archivo).replace("\\", "/")
            
            # Agregar la imagen al grupo del ID correspondiente
            if person_id not in imgs_por_id:
                imgs_por_id[person_id] = []
            imgs_por_id[person_id].append(img_rel)

    # Eliminar IDs con menos de 2 imágenes (no pueden generar pares positivos)
    imgs_por_id = {pid: sorted(imgs) for pid, imgs in imgs_por_id.items() if len(imgs) >= 2}
    
    ids_personas = sorted(imgs_por_id.keys())

    with open(output_txt, "w", encoding="utf-8") as f:
        # Positivos (misma persona = mismo ID, diferentes imágenes)
        for person_id in ids_personas:
            for img1, img2 in itertools.combinations(imgs_por_id[person_id], 2):
                f.write(f"{img1} {img2} 1\n")

        # Negativos (personas diferentes = IDs diferentes)
        for i, id1 in enumerate(ids_personas):
            for id2 in ids_personas[i + 1:]:
                for img1 in imgs_por_id[id1]:
                    for img2 in imgs_por_id[id2]:
                        f.write(f"{img1} {img2} 0\n")

    total_imgs = sum(len(v) for v in imgs_por_id.values())
    print(f"\n{output_txt} generado exitosamente!")
    print(f"  - {len(ids_personas)} personas (IDs únicos)")
    print(f"  - {total_imgs} imágenes totales")
    
    # Debug: mostrar qué imágenes se agruparon por ID
    print("\nAgrupación por ID de persona:")
    for pid in sorted(imgs_por_id.keys()):
        print(f"  ID {pid}: {len(imgs_por_id[pid])} imágenes")
        for img in imgs_por_id[pid]:
            print(f"    - {img}")


# Ejemplo de uso:
if __name__ == "__main__":
    root_dir = "./data"  # Carpeta raíz donde están crd_0811f, crd_0813f, etc.
    
    # Obtener automáticamente todos los usuarios (carpetas que empiezan con "crd_")
    users = [d for d in os.listdir(root_dir) 
             if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("crd_")]
    
    print(f"Usuarios encontrados: {users}\n")
    
    generar_pairs_txt_lfw_from_users(root_dir, users, "pairs.txt")