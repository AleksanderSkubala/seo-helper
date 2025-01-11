import os
from openai import OpenAI
import pandas as pd
import json


def use_model(model_id, messages):
  client = OpenAI()

  completion = client.chat.completions.create(
    model=model_id,
    messages=messages
  )

  return completion.choices[0].message.content


def create_chain_of_thought(max_messages=50):
  DATA_DIR = "data"
  FINE_TUNING_DATA_FILE = os.path.join(DATA_DIR, "fine_tuning_data.jsonl")

  cot_messages = []
  cot_messages.append({"role": "system", "content": "You are an editor in a high-end Polish magazine like Vogue or Przekrój. You are writing an article about the topics provided by the user.\nWrite in a high-end and sophisticated, but not formal or academic style, rather a poetic/artistic/journalist style. Your text must be written in Polish!\nIn your answer include the proposed title and the content of the article divided with a triple hash (###). Don't write anything else in your answer."})

  with open(FINE_TUNING_DATA_FILE, 'r', encoding='utf-8') as file:
    for line in file:
      example = json.loads(line)
      messages = example.get('messages', [])
      if len(messages) >= 3:
        cot_messages.append(messages[1])
        cot_messages.append(messages[2])
  return cot_messages[:max_messages+1]


if __name__ == "__main__":
  OPENAI_API_KEY = '###'
  os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

  cot_base = create_chain_of_thought()
  cot_base.append({"role": "user", "content": "Napisz artykuł na podane tematy: 1. Krytyka adaptacji: Serial Netflixa na podstawie \"Sto lat samotności\" Gabriela Garcíi Márqueza nie spełnia oczekiwań, stając się tasiemcem w stylu telenowel, mimo starannej inscenizacji.\n\n2. Temat powieści: \"Sto lat samotności\" przedstawia historię rodziny Buendiów w fikcyjnym miasteczku Macondo, oferując refleksję na temat cywilizacji, pokoleń oraz więzi rodzinnych, w tym tragicznych aspektów kazirodztwa.\n\n3. Proces twórczy: Márquez stworzył powieść po długim okresie pracy, wykorzystując swoje osobiste doświadczenia i tradycje narracyjne z dzieciństwa, co wprowadziło lokalny kontekst do jego pisarstwa.\n\n4. Sukces literacki: Książka zdobyła międzynarodowe uznanie, a Gabriel García Márquez otrzymał Nagrodę Nobla w 1982 roku, co spowodowało masowe tłumaczenia i adaptacje.\n\n5. Elementy realizmu magicznego: Powieść łączy rzeczywistość z nieprawdopodobnymi wydarzeniami, które odzwierciedlają latynoamerykański kontekst kulturowy oraz społeczno-polityczną rzeczywistość regionu.\n\n6. Tematy rodziny i przemocy: Serial ukazuje powtarzające się schematy rodzinne oraz przemoc, co prowadzi do zniszczenia społeczności Macondo i zwiastuje jej upadek.\n\n7. Odmienność adaptacji: W porównaniu do \"Stu lat samotności\", film \"Pedro Páramo\" jest bardziej wizualny i poetycki, a jego tematy dotyczą zarówno pamięci, jak i zapomnienia, tworząc związki z późniejszymi dziełami Márqueza.\n\n8. Refleksja nad samotnością: W \"Stu latach samotności\" samotność (soledad) jest przedstawiana jako coś więcej niż tylko brak towarzystwa – to stan samostanowienia i poczucia upływu czasu, a przestrzeń na refleksję jest istotna w narracji.\n\nArtykuł powinien zawierać 500-600 słów."})
  
  model_id = "gpt-4o-mini"
  print(use_model(model_id, cot_base))