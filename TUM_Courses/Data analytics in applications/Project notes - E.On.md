
## First experiment notes:
### Deberta_finetuned_pii

link: https://huggingface.co/lakshyakh93/deberta_finetuned_pii

**Example usage**
```python 
from transformers import AutoTokenizer, AutoModelForTokenClassification

from transformers import pipeline

gen = pipeline("token-classification", "lakshyakh93/deberta_finetuned_pii", device=-1)

text = " Contract number 404149058. The Zirme/Rogner family Henkring 06 58954 Freising. Dear Sir or Madam, we sent you an email on July 17, 2023 intentionally, and unfortunately, after almost 1.5 months, we still haven't received a response. Please write again today. An error occurred in the last bill dated June 29, 2023, namely: The meter reading is incorrect. On June 18, 2023, our actual meter reading was 13,962 m3. According to your bill, it was 14,042 m3. That's a difference of 80 m3 (i.e., 80 m3 too much). We offer to convert everything for you. Thank you very much. Best regards, The Zirme/Rogner family"  

output = gen(text, aggregation_strategy="first")

print(output)
```

Pre-trained model `deberta_finetuned_pii` performs well on simple texts like "Hello my name is Andrew Forest and i am 50 years old.", correctly identifying Fore- and lastname. However, for longer texts translated form german (E.On dataset), it has a very hard time identifying the correct PII. The original german texts yields even worse performance, making the model guess completely. 

Pros: 
- Performs well on small input texts in english 
- Inference can be run on CPU

Cons:
- Doesn't not scale with the size of the text. 
- Terrible in german
### obi/Medical-Note-Deidentification

link: https://huggingface.co/obi/deid_roberta_i2b2

This model performs much better on the longer texts that E.On has provided, giving a clear indication of what information is PII and not. As with the example in this figure: 

**Example usage**
![[Pasted image 20250522110208.png]]
##### Comparison of german vs. english translated performance
<div style="display: flex;">
  <img src="Pasted image 20250522110534.png" width="350" style="margin-right: 10px;"/>
  <img src="Pasted image 20250522110633.png" width="350"/>
</div>

We see that the original german email yields slightly different results compared to the translated english version. The german version is overly sensitive and thinks that "iban" is a location. This is not the case the the translated english version. The english version also manages to identify that "Kartoffelbank" is of importance, however, logically it is identified as a hospital and not a bank. This is because of the pre-trained model which is trained on hospital records. 

Pros: 
+ Performs okay on both german and english emails (must be tested more).
+ Millions of downloads - updated community 
+ Easily interpreted and well described. 
+ Inference can be run on CPU

Cons: 
- Heavily pre-trained on specifically medical information (i.e. confusing banks with hospitals). 

### PII-Model-Phi3-Mini ❌

link: https://huggingface.co/ab-ai/PII-Model-Phi3-Mini

With CPU: Crashes after using all available RAM. 

### ab-ai/pii_model 

link: https://huggingface.co/ab-ai/pii_model

**Example usage**
```python 
# Load model directly
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("ab-ai/pii_model")

model = AutoModelForTokenClassification.from_pretrained("ab-ai/pii_model")

text = "Hallo EON, Bitte erstatten Sie auf folgendes Konto zurÃ¼ck Chantal Roskoth Iban: DE28263410720363784400 bei der Kartoffelbank. Mit freundlichen GrÃ¼ÃŸen Chantal Roskoth"  

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():

outputs = model(**inputs)

# Get predicted label IDs

logits = outputs.logits

predicted_class_ids = torch.argmax(logits, dim=-1).squeeze().tolist()

# Get label names

labels = model.config.id2label

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
```

This model performs well both on german and english for small texts. Atleast for the examples. 

Output: [('FIRSTNAME', 'chantal'), ('LASTNAME', 'roskoth'), ('IBAN', 'de28263410720363784400'), ('STATE', 'kartoffelbank.'), ('FIRSTNAME', 'chantal'), ('LASTNAME', 'roskoth')]

However, for long texts, it performs very bad on the german texts, but okay on the english translated texts. 

German text results: [('ACCOUNTNUMBER', '404149058'), ('COMPANYNAME', 'z'), ('ZIPCODE', '58954'), ('STATE', 'fr'), ('TIME', '17.07.23'), ('IPV4', '29.'), ('PHONENUMBER', '06.2023'), ('TIME', '18.06.23'), ('ZIPCODE', '13962'), ('ZIPCODE', '14042'), ('COMPANYNAME', 'zirme/rogner')]

English text results: [('ACCOUNTNUMBER', '404149058.'), ('COMPANYNAME', 'zirme/rogner'), ('BUILDINGNUMBER', '06'), ('ZIPCODE', '58954'), ('STATE', 'freising.'), ('DATE', 'july17,2023'), ('DATE', 'june29,2023'), ('DATE', 'june18,2023,'), ('AMOUNT', '13,962m3.'), ('AMOUNT', '14,042m3.'), ('HEIGHT', '80m3'), ('COMPANYNAME', 'zirme/rogner')]

Pros: 
- Easy to interpret.
- Runs inference on CPU.
- good performance for english texts.
- Okay performance on shorter german texts.

Cons: 
- Bad at longer german texts.
- Requires helper functions to better understand the results.  


