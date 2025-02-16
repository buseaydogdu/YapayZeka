import pandas as pd
print(pd.__version__)

# arac_satis_verileri = {
#     "arabalar":["bmw","ferrari","togg"],
#     "satis_rakamlari":[200,10,150],
#     "yillari":[2000,1997,2006]
# } 

arac_satis_verileri = pd.read_csv("ornekCSVdosyasi.csv")

pandas_ile_olusturulan_dizi = pd.DataFrame(arac_satis_verileri)
print(pandas_ile_olusturulan_dizi )