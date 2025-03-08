import re
txt1 = "Ahmet al renkli bir şal aldı."
txt2 = "Mehmet kırmızı top getirdi."

bulunan1 = re.search("al", txt1)
print(type(bulunan1))
print(bulunan1.span())
print(bulunan1.start())
print(bulunan1.end())
