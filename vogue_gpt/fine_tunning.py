import os
from openai import OpenAI
import pandas as pd
import random
import json


def summarise_content(content):
  client = OpenAI()

  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Czy mógłbyś podsumować podany tekst do kilku najważniejszych punktów, tak jakby miał z nich następnie zostać ten tekst odtworzony?\n===Artykuł=== \nCo jest trudniejsze – zniknąć ze sceny, pozostawiając po sobie gwiezdny pył, czy wrócić z fanfarami, cały w pudrze i brokacie, ze złocistą poświatą na skórze, piórami we włosach oraz wszędzie tam, gdzie ich obecność da się wytłumaczyć wymogami artystycznymi? Singiel Eltona Johna „The Bitch is Back” w tym roku obchodzi 50. urodziny. Elton Johnw operowym stylu świętuje właśnie premierę musicalu„The Devil Wears Prada”na londyńskim West Endzie – sam skomponował muzykę (zrozumiałe),Mirandę Priestlygra Vanessa Williams – kobieta rakieta, dobra aktorka, doskonała wokalistka lub odwrotnie, zależy, co cenimy bardziej. Czy Elton pozwoli jej zabłysnąć, podwinie pod siebie lisią kitę, czy raczej ukradnie cały show, jak w zwyczaju mają osobniki gatunku homo sapiens, podgatunekbitch, tłumaczenie niepotrzebne? „The Bitch is Back”,drugi singiel z płyty „Carobou”, totypowy Elton John– rockandrollowy, wysokooktanowy, od premiery brzmi lekko retro, więc źle zestarzeć się nie może. Wymarzony numer do „Tańca z gwiazdami”, do chwalenia się dużym budżetem na ortodontę i dermatologa, niezbędnym wobec poczucia rytmu na poziomie kapibary. Kapibara, wybacz. Tekst napisał Bernie Taupin, po tym jak jego żona Maxine słusznie zasugerowała, że John to osobowość dwubiegunowa. Co najmniej. Raz ma chłopak doła, nie widzi dla siebie miejsca tu i teraz, o przyszłości nie wspominając, częściej jednak prze do przodu niczym koreański czołg, po drodze rozjeżdża wszystkich w poczuciu, że tam, gdzie pojawia się gwiazda dla zwyczajnych, normalnych i przede wszystkim przeciętnych, świata już nie starcza. Ten fenomen Taupin tłumaczy tak: “I'm a bitch, I'm a bitch, oh, the bitch is backStone-cold sober, as a matter of factI can bitch, I can bitch 'cause I'm better than youIt's the way that I move, the things that I do, oh-oh-oh” Między najnowszą premierą na West Endzie i wszystkimi poprzednimi od 1968 roku Elton John zdążył zaliczyć setki kryzysów, zmian tożsamości i nie tylko. Tracił wiarę w siebie, tak tyci, tyci, i odzyskiwał z nawiązką;uzależniał się strasznie, szorował po dnie, ale szedł na odwyk i było po nałogu; nabierał ciała, pamiętamy, ale migiem chudł w stylu, który na przełomie stuleci nadwiślańska prasa klasy B uwielbiała nazywać „spektakularnym” i głośno biła brawo. Gdy tak puchł, po czym znikał i po roku wracał wylaszczony, nie było jeszcze ozempicu, mimo to ani razu nie bredził o tęczowej diecie – w poniedziałek jemy tylko zielone, we wtorek żółte…Christina Aguileratakiego trucia nie potrafi nam oszczędzić, dlatego słuchamy jej mniej cierpliwie. Na YouTubie między wszystkimi wersjami „The Bitch is Back” – solo w studiu, solo na stadionie, w ducie zTiną Turneri z Rihanną – zaplątał się filmik, w którym Elton wspomina najciekawsze wdzianka, jakimi raczył nasze oczęta. Nad zrobieniem z utalentowanego wokalisty widowiska uczciwie napracowali sięBob Macki, potemGianni Versace, jeden raz nawetGiorgio Armani. Kto wpadł na pomysł, by wcisnąć go w sweterek typu trzmiel – szerokie czarne i żółte pasy – a kto namówił na koncert w kostiumie Kaczora Donalda, Elton oczywiście dziś nie pamięta. Rasowabitchnigdy niczego nie żałuje. I nie musi się z niczego tłumaczyć. Never ever. \n==="
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": "1. 50-lecie singla: Elton John obchodzi 50. rocznicę wydania singla „The Bitch is Back”.\n2. Premier musicalu: Kompozytor muzyki do musicalu „The Devil Wears Prada” na londyńskim West Endzie, w którym Vanessa Williams gra główną rolę.\n3. Kariera Eltona Johna: Autor tekstu, Bernie Taupin, porusza w nim złożoną osobowość Eltona, jego wzloty i upadki, uzależnienia oraz odnajdywanie siebie.\n4. Styl i wizerunek: Elton John znany jest z spektakularnych strojów i występów, które były wynikiem współpracy z uznanymi projektantami mody.\n5. Fenomen osobowości: Opisuje, jak Elton przyciąga uwagę, zmieniając się i odnajdując nowe oblicze na scenie oraz porównuje go do „bitch”, co sugeruje silną, bezkompromisową postawę. \n6. Historia uzależnień: Podkreślono, że Elton zmagał się z uzależnieniami, a jego droga do zdrowia była pełna wyzwań.\n7. Kulturowy wpływ: W artykule zawarto odniesienia do sposobu, w jaki Elton John kształtował swoje życie artystyczne i publiczny wizerunek, jednocześnie pozostając głęboko niedoskonałą, ale autentyczną osobą."
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": f"Czy mógłbyś podsumować podany tekst do kilku najważniejszych punktów, tak jakby miał z nich następnie zostać ten tekst odtworzony?\n===Artykuł===\n{content}\n==="
          }
        ]
      }
    ],
    response_format={
      "type": "text"
    },
    temperature=1,
    max_completion_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  if response.choices[0].message.content is not None:
    return response.choices[0].message.content
  else:
    return None


def clear_content(content):
  client = OpenAI()

  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Czy mógłbyś wyczyścić podany artykuł? Zależy mi głównie, żebyś:\n1. dodał spacje tam gdzie ich brakuje\n2. dokończył słowa, którym brakuje liter\n3. usunął niedokończone zdania.\nOdpowiedz samym poprawionym tekstem - nie pisz nic więcej.\n\n===Artykuł=== \nCo jest trudniejsze – zniknąć ze sceny, pozostawiając po sobie gwiezdny pył, czy wrócić z fanfarami, cały w pudrze i brokacie, ze złocistą poświatą na skórze, piórami we włosach oraz wszędzie tam, gdzie ich obecność da się wytłumaczyć wymogami artystycznymi? Singiel Eltona Johna „The Bitch is Back” w tym roku obchodzi 50. urodziny. Elton Johnw operowym stylu świętuje właśnie premierę musicalu„The Devil Wears Prada”na londyńskim West Endzie – sam skomponował muzykę (zrozumiałe),Mirandę Priestlygra Vanessa Williams – kobieta rakieta, dobra aktorka, doskonała wokalistka lub odwrotnie, zależy, co cenimy bardziej. Czy Elton pozwoli jej zabłysnąć, podwinie pod siebie lisią kitę, czy raczej ukradnie cały show, jak w zwyczaju mają osobniki gatunku homo sapiens, podgatunekbitch, tłumaczenie niepotrzebne? „The Bitch is Back”,drugi singiel z płyty „Carobou”, totypowy Elton John– rockandrollowy, wysokooktanowy, od premiery brzmi lekko retro, więc źle zestarzeć się nie może. Wymarzony numer do „Tańca z gwiazdami”, do chwalenia się dużym budżetem na ortodontę i dermatologa, niezbędnym wobec poczucia rytmu na poziomie kapibary. Kapibara, wybacz. Tekst napisał Bernie Taupin, po tym jak jego żona Maxine słusznie zasugerowała, że John to osobowość dwubiegunowa. Co najmniej. Raz ma chłopak doła, nie widzi dla siebie miejsca tu i teraz, o przyszłości nie wspominając, częściej jednak prze do przodu niczym koreański czołg, po drodze rozjeżdża wszystkich w poczuciu, że tam, gdzie pojawia się gwiazda dla zwyczajnych, normalnych i przede wszystkim przeciętnych, świata już nie starcza. Ten fenomen Taupin tłumaczy tak: “I'm a bitch, I'm a bitch, oh, the bitch is backStone-cold sober, as a matter of factI can bitch, I can bitch 'cause I'm better than youIt's the way that I move, the things that I do, oh-oh-oh” Między najnowszą premierą na West Endzie i wszystkimi poprzednimi od 1968 roku Elton John zdążył zaliczyć setki kryzysów, zmian tożsamości i nie tylko. Tracił wiarę w siebie, tak tyci, tyci, i odzyskiwał z nawiązką;uzależniał się strasznie, szorował po dnie, ale szedł na odwyk i było po nałogu; nabierał ciała, pamiętamy, ale migiem chudł w stylu, który na przełomie stuleci nadwiślańska prasa klasy B uwielbiała nazywać „spektakularnym” i głośno biła brawo. Gdy tak puchł, po czym znikał i po roku wracał wylaszczony, nie było jeszcze ozempicu, mimo to ani razu nie bredził o tęczowej diecie – w poniedziałek jemy tylko zielone, we wtorek żółte…Christina Aguileratakiego trucia nie potrafi nam oszczędzić, dlatego słuchamy jej mniej cierpliwie. Na YouTubie między wszystkimi wersjami „The Bitch is Back” – solo w studiu, solo na stadionie, w ducie zTiną Turneri z Rihanną – zaplątał się filmik, w którym Elton wspomina najciekawsze wdzianka, jakimi raczył nasze oczęta. Nad zrobieniem z utalentowanego wokalisty widowiska uczciwie napracowali sięBob Macki, potemGianni Versace, jeden raz nawetGiorgio Armani. Kto wpadł na pomysł, by wcisnąć go w sweterek typu trzmiel – szerokie czarne i żółte pasy – a kto namówił na koncert w kostiumie Kaczora Donalda, Elton oczywiście dziś nie pamięta. Rasowabitchnigdy niczego nie żałuje. I nie musi się z niczego tłumacz. Never ev. Przy tym wszystkim Elton\n==="
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": "Co jest trudniejsze – zniknąć ze sceny, pozostawiając po sobie gwiezdny pył, czy wrócić z fanfarami, cały w pudrze i brokacie, ze złocistą poświatą na skórze, piórami we włosach oraz wszędzie tam, gdzie ich obecność da się wytłumaczyć wymogami artystycznymi? Singiel Eltona Johna „The Bitch is Back” w tym roku obchodzi 50. urodziny. Elton Johnw operowym stylu świętuje właśnie premierę musicalu „The Devil Wears Prada” na londyńskim West Endzie – sam skomponował muzykę (zrozumiałe), Mirandę Priestly gra Vanessa Williams – kobieta rakieta, dobra aktorka, doskonała wokalistka lub odwrotnie, zależy, co cenimy bardziej. Czy Elton pozwoli jej zabłysnąć, podwinie pod siebie lisią kitę, czy raczej ukradnie cały show, jak w zwyczaju mają osobniki gatunku homo sapiens, podgatunek bitch, tłumaczenie niepotrzebne? „The Bitch is Back”, drugi singiel z płyty „Carobou”, to typowy Elton John– rockandrollowy, wysokooktanowy, od premiery brzmi lekko retro, więc źle zestarzeć się nie może. Wymarzony numer do „Tańca z gwiazdami”, do chwalenia się dużym budżetem na ortodontę i dermatologa, niezbędnym wobec poczucia rytmu na poziomie kapibary. Kapibara, wybacz. Tekst napisał Bernie Taupin, po tym jak jego żona Maxine słusznie zasugerowała, że John to osobowość dwubiegunowa. Co najmniej. Raz ma chłopak doła, nie widzi dla siebie miejsca tu i teraz, o przyszłości nie wspominając, częściej jednak prze do przodu niczym koreański czołg, po drodze rozjeżdża wszystkich w poczuciu, że tam, gdzie pojawia się gwiazda dla zwyczajnych, normalnych i przede wszystkim przeciętnych, świata już nie starcza. Ten fenomen Taupin tłumaczy tak: “I'm a bitch, I'm a bitch, oh, the bitch is back. Stone-cold sober, as a matter of fact, I can bitch, I can bitch 'cause I'm better than you. It's the way that I move, the things that I do, oh-oh-oh” Między najnowszą premierą na West Endzie i wszystkimi poprzednimi od 1968 roku Elton John zdążył zaliczyć setki kryzysów, zmian tożsamości i nie tylko. Tracił wiarę w siebie, tak tyci, tyci, i odzyskiwał z nawiązką; uzależniał się strasznie, szorował po dnie, ale szedł na odwyk i było po nałogu; nabierał ciała, pamiętamy, ale migiem chudł w stylu, który na przełomie stuleci nadwiślańska prasa klasy B uwielbiała nazywać „spektakularnym” i głośno biła brawo. Gdy tak puchł, po czym znikał i po roku wracał wylaszczony, nie było jeszcze ozempicu, mimo to ani razu nie bredził o tęczowej diecie – w poniedziałek jemy tylko zielone, we wtorek żółte… Christina Aguileratakiego trucia nie potrafi nam oszczędzić, dlatego słuchamy jej mniej cierpliwie. Na YouTubie między wszystkimi wersjami „The Bitch is Back” – solo w studiu, solo na stadionie, w ducie z Tiną Turner i z Rihanną – zaplątał się filmik, w którym Elton wspomina najciekawsze wdzianka, jakimi raczył nasze oczęta. Nad zrobieniem z utalentowanego wokalisty widowiska uczciwie napracowali się Bob Macki, potem Gianni Versace, jeden raz nawet Giorgio Armani. Kto wpadł na pomysł, by wcisnąć go w sweterek typu trzmiel – szerokie czarne i żółte pasy – a kto namówił na koncert w kostiumie Kaczora Donalda, Elton oczywiście dziś nie pamięta. Rasowa bitch nigdy niczego nie żałuje. I nie musi się z niczego tłumaczyć. Never ever.\n==="
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": f"Czy mógłbyś wyczyścić podany artykuł? Zależy mi głównie, żebyś:\n1. dodał spacje tam gdzie ich brakuje\n2. dokończył słowa, którym brakuje liter\n3. usunął niedokończone zdania.\nOdpowiedz samym poprawionym tekstem - nie pisz nic więcej.\n\n===Artykuł===\n{content}\n==="
          }
        ]
      }
    ],
    response_format={
      "type": "text"
    },
    temperature=1,
    max_completion_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  if response.choices[0].message.content is not None:
    return response.choices[0].message.content
  else:
    return None


def prepare_data_for_fine_tuning(input_csv, output_json, num_samples=75):
  # Load data from CSV
  df = pd.read_csv(input_csv)
  
  # Pick random K-number of examples
  sampled_df = df.sample(n=num_samples, random_state=42)
  
  # Prepare data in OpenAI's fine-tuning prompt format
  i = 1
  fine_tuning_data = []
  for _, row in sampled_df.iterrows():
    fine_tuning_data.append({
      "messages": [
        {"role": "system", "content": "You are an editor in a high-end Polish magazine like Vogue or Przekrój. You are writing an article about the topics provided by the user.\nWrite in a high-end and sophisticated, but not formal or academic style, rather a poetic/artistic/journalist style. Your text must be written in Polish!\nIn your answer include the proposed title and the content of the article divided with a triple hash (###). Don't write anything else in your answer."},
        {"role": "user", "content": f"Napisz artykuł na podane tematy: {summarise_content(row['content'])}"},
        {"role": "assistant", "content": f"{clear_content(row['heading'])}###{clear_content(row['content'])}"}
      ]
    })

    # Export data to JSONL file
    with open(output_json, 'w') as f:
      for entry in fine_tuning_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"{i}/{num_samples} already done")
    i += 1


def fine_tune_model(fine_tuning_data_file, output_file, num_epochs):
  client = OpenAI()

  file_obj = client.files.create(
    file=open(fine_tuning_data_file, "rb"),
    purpose="fine-tune"
  )
  print(file_obj)

  job_obj = client.fine_tuning.jobs.create(
    training_file=file_obj.id,
    model="gpt-4o-mini-2024-07-18",
    method={
      "type": "supervised",
      "supervised": {
        "hyperparameters": {
          "n_epochs": "auto" if num_epochs is None else num_epochs
        }
      }
    }
  )
  # print(job_obj)
  print(client.fine_tuning.jobs.list())

  with open(output_file, "w") as f:
    f.write(f"{job_obj}")


def retrieve_fine_tuned_model(job_id):
  client = OpenAI()

  retrieved_obj = client.fine_tuning.jobs.retrieve(job_id)

  # if status not 'running'
  return retrieved_obj.fine_tuned_model


def use_fine_tunned_model(model_id, messages):
  client = OpenAI()

  completion = client.chat.completions.create(
    model=model_id,
    messages=messages
  )

  return completion.choices[0].message.content


def fine_tunning_process():
  DATA_DIR = "data"
  LOG_DIR = "logs"
  SCRAPED_DATA_FILE = os.path.join(DATA_DIR, "articles_metadata.csv")
  FINE_TUNING_DATA_FILE = os.path.join(DATA_DIR, "fine_tuning_data_5.jsonl")
  OUTPUT_FILE = os.path.join(LOG_DIR, "fine_tuning_job_info_5.txt")

  prepare_data_for_fine_tuning(SCRAPED_DATA_FILE, FINE_TUNING_DATA_FILE, num_samples=10)
  fine_tune_model(FINE_TUNING_DATA_FILE, OUTPUT_FILE, num_epochs=1)


def usage_process(ftjob_id, check=False):
  messages = [
    {"role": "system", "content": "You are an editor in a high-end Polish magazine like Vogue or Przekrój. You are writing an article about the topics provided by the user.\nWrite in a high-end and sophisticated, but not formal or academic style, rather a poetic/artistic/journalist style. Your text must be written in Polish!\nIn your answer include the proposed title and the content of the article divided with a triple hash (###). Don't write anything else in your answer."},
    {"role": "user", "content": "Napisz artykuł na podane tematy:\n1. Dekadentyzm: Czym odznacza się dekadentyzm w kulturze i filozofii?\n2. Dekandentyzm XXI wieku: Czym jest dekadentyzm w XXI wieku i jakie są jego przejawy w kulturze popularnej i internecie? Jak wobec dekadentyzmu zachowuje się człowiek współczesny i czy jest na niego podatny\n3. Dekadentyzm w kulturze nowoczesnej: Jakie są przykłady dekadentyzmu w kulturze nowoczesnej, w tym w literaturze, filmie, muzyce i sztuce?\n4. Dekadentyzm w Polsce: Jakie są przykłady dekadentyzmu w polskiej kulturze i literaturze? Jakie są polskie interpretacje dekadentyzmu i jakie są ich różnice w porównaniu z dekadentyzmem zachodnim?\n5. Dekadentyzm w internecie: Jakie są przykłady dekadentyzmu w internecie i jakie są jego przejawy w mediach społecznościowych, memach i kulturze popularnej?"},
  ]

  model_id = retrieve_fine_tuned_model(ftjob_id)
  if check:
    print(model_id)
  if not check:
    print(use_fine_tunned_model(model_id, messages))


if __name__ == "__main__":
  OPENAI_API_KEY = '###'
  os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

  # fine_tunning_process()
  usage_process("ftjob-nmfnx4PWlN8WtVj0cC8IH5lK", check=False)