import re
txt1 = "Ahmet al renkli bir şal aldı."
txt2 = "Mehmet kırmızı top getirdi."

bulunan1 = re.findall("al", txt1)
bulunan1 = re.findall("a", txt1)
bulunan1 = re.findall("a|A", txt1)  
# | veya analmındadır.
print(bulunan1)