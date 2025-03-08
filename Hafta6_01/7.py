import re
txt1 = "Ahmet al renkli bir şal aldı."
txt2 = "Mehmet kırmızı top getirdi."

xxx = "Ahmet al renkli bir şal aldı."

cumleler = [txt1,txt2,xxx]

for a in cumleler:
    kelimeleri = re.split("\s", a)
    print(kelimeleri) 