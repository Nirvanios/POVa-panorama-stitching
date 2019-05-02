ZPO - Skládání fotografií


Spojování více obrázku do jednoho panorama.


použití:

main.py [-h] --folder FOLDER --img IMG --dest DEST [--out OUT]
               [--kp_detector {SIFT,SURF}] [--pano_type {HOMOGRAPHY,AFFINE}]
               [--blender {None, weight, graph_cut}][--cyl_wrap] [--debug]

nepovinné parametry:

  -h, --help            Zobrazí nápovědu
  --folder FOLDER       Cesta k obrázkům pro spojování
  --img IMG             Cesta k hlavnímu obrázku na který se budou ostatní napojovat
  --dest DEST           Cesta v výstupnímu panoramatu
  --out OUT             Umístění výstupních debug informací
  --kp_detector {SIFT,SURF}
                        Výběr detektoru významných bodů
  --pano_type {HOMOGRAPHY,AFFINE}
                        Typ transformace afiní nebo homografie
  --cyl_wrap            Zapne projekci na válec
  --debug               Zapne debug výpisy
