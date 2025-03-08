""" 📌
Başlangıç seviyesinde harf, rakam ve kelime bulma
Orta seviyede e-posta, telefon, URL gibi yapılar
İleri seviyede Lookahead, Lookbehind ve özel patternler
Bu örneklerle regex pratiği yaparak çok güçlü metin işleme teknikleri geliştirebilirsiniz! 🚀
"""

import re

# Sadece rakamları bulma
print(re.findall(r"\d+", "Benim 2 kedim,5 kuşum ve 10 köpeğim var."))  

# Sadece harfleri bulma
print(re.findall(r"[a-zA-Z]+", "Merhaba123, dünya!"))  

# Bir string’in belirli bir kelimeyle başlayıp başlamadığını kontrol etme
print(re.match(r"^Merhaba", "Merhaba dünya"))  # Eşleşme var
print(re.match(r"^Merhaba", "Dünya merhaba"))  # None (Eşleşme yok)

# Bir string’in belirli bir kelimeyle bitip bitmediğini kontrol etme
print(re.search(r"dünya!$", "Merhaba dünya!"))  # Eşleşme var
print(re.search(r"dünya!$", "Dünya merhaba."))  # None (Eşleşme yok)

# E-posta adreslerini bulma
print(re.findall(r"\w+@\w+\.\w+", "Mailim: test@example.com, diğeri: deneme@site.net"))  

# Telefon numaralarını bulma
print(re.findall(r"\d{3}-\d{3}-\d{4}", "Telefonlar: 555-123-4567, 123-456-7890"))  

# Büyük harfle başlayan kelimeleri bulma
print(re.findall(r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\b", "Ali ve Ahmet sinemaya gitti."))  

# Sadece kelimeleri almak (noktalama işaretleri hariç)
print(re.findall(r"\b\w+\b", "Merhaba, nasılsın? Bugün hava çok güzel!"))  

# Boşluk karakterleriyle ayrılmış sayıları bulma
print(re.findall(r"\d+\s+\d+", "25 30 ve 100 200 sayıları var."))  

# Hexadecimal renk kodlarını bulma
print(re.findall(r"#[0-9A-Fa-f]{6}", "Renk kodları: #FF5733, #123ABC, #ZXY123")) 
