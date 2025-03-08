# ör-1: findall ile metin içindekileri alma
import re
txt1 = "Ahmet al renkli bir şal aldı."
txt2 = "Mehmet kırmızı bir top getirdi."
print(re.findall("al", txt1))
print(re.findall("al", txt2))

# ör-2: search ile metin içinde arama
import re
txt = "Ahmet al renkli bir şal aldı."
aa = re.search("\s", txt)
print("İlk boşluğun bulunduğu yer:", aa.start())
bb = re.search("Mehmet", txt); print(bb)
print(re.search("şal", txt))


