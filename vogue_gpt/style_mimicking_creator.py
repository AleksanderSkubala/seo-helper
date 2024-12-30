import os
from openai import OpenAI
from typing import List, Dict

# ================================
# Configuration
# ================================

# Set your OpenAI API key
OPENAI_API_KEY = '###'

# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI()

# ================================
# Define Functions for Each Stage
# ================================

def confirm_receipt(sample_text: str) -> str:
    """
    Confirms receipt of the sample text.
    """
    prompt = f"""
    You are an analytical assistant tasked with extracting stylistic features from a sample article.

    Please confirm receipt and readiness to analyze the text for style extraction.

    ---
    {sample_text}
    ---
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2)
    confirmation = response.choices[0].message.content.strip()
    return confirmation

def general_analysis(sample_text: str) -> str:
    """
    Analyzes and describes the overall writing style.
    """
    prompt = f"""
    Based on the provided sample article, analyze and describe the overall writing style. Consider aspects such as sophistication, readability, and engagement level. Provide a concise summary highlighting key stylistic attributes.
    
    ---
    {sample_text}
    ---
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2)
    analysis = response.choices[0].message.content.strip()
    return analysis

def lexical_analysis(sample_text: str) -> str:
    """
    Analyzes the vocabulary used in the sample article.
    """
    prompt = f"""
    Analyze the vocabulary used in the sample article. Highlight aspects such as:
    - Vocabulary richness and diversity
    - Use of advanced or specialized terminology
    - Preference for specific types of words (e.g., descriptive adjectives, precise verbs)
    
    Provide examples of notable word choices.
    
    ---
    {sample_text}
    ---
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2)
    analysis = response.choices[0].message.content.strip()
    return analysis

def syntactic_analysis(sample_text: str) -> str:
    """
    Analyzes sentence structures, complexity, and variety.
    """
    prompt = f"""
    Examine the sentence structures in the sample article. Discuss:
    - Sentence length and complexity
    - Use of varied sentence structures (e.g., compound, complex, simple)
    - Punctuation usage and its effect on pacing
    
    Provide specific examples to illustrate these points.
    
    ---
    {sample_text}
    ---
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2)
    analysis = response.choices[0].message.content.strip()
    return analysis

def structural_analysis(sample_text: str) -> str:
    """
    Analyzes the structural organization of the sample article.
    """
    prompt = f"""
    Analyze the structural organization of the sample article. Consider:
    - Overall layout (e.g., number of sections, headings)
    - Logical flow of ideas
    - Use of transitions between paragraphs and sections
    
    Provide a summary of the structural characteristics.
    
    ---
    {sample_text}
    ---
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2)
    analysis = response.choices[0].message.content.strip()
    return analysis

def tone_and_register_analysis(sample_text: str) -> str:
    """
    Assesses the tone and register of the sample article.
    """
    prompt = f"""
    Assess the tone and register of the sample article. Address:
    - Formality level
    - Emotional undertones (e.g., neutral, persuasive, inspirational)
    - Consistency in tone throughout the text
    
    Illustrate your analysis with examples from the article.
    
    ---
    {sample_text}
    ---
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2)
    analysis = response.choices[0].message.content.strip()
    return analysis

def figurative_language_analysis(sample_text: str) -> str:
    """
    Identifies and analyzes figurative language and rhetorical devices.
    """
    prompt = f"""
    Identify and analyze the use of figurative language and rhetorical devices in the sample article. Highlight:
    - Types of figurative language used (e.g., metaphors, similes)
    - Rhetorical devices employed (e.g., alliteration, parallelism)
    - Their effectiveness in enhancing the text
    
    Provide examples from the article.
    
    ---
    {sample_text}
    ---
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2)
    analysis = response.choices[0].message.content.strip()
    return analysis

def stylistic_nuances_analysis(sample_text: str) -> str:
    """
    Captures unique stylistic elements specific to the text.
    """
    prompt = f"""
    Examine any unique stylistic nuances present in the sample article that contribute to its distinctive voice. Consider elements such as:
    - Use of anecdotes or personal stories
    - Repetition for emphasis
    - Unique formatting or structural choices
    
    Detail how these nuances affect the overall style.
    
    ---
    {sample_text}
    ---
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2)
    analysis = response.choices[0].message.content.strip()
    return analysis

def compile_style_guide(analyses: Dict[str, str]) -> str:
    """
    Compiles all extracted stylistic analyses into a comprehensive style guide.
    """
    prompt = f"""
    Based on your analyses from the previous stages, compile a detailed style guide that encapsulates all the extracted linguistic features of the sample article. The style guide should include:
    
    1. **Vocabulary**
       - Richness and diversity
       - Preferred word types and examples
    
    2. **Sentence Structure**
       - Complexity and variety
       - Punctuation usage
    
    3. **Organization**
       - Structural layout
       - Flow and transitions
    
    4. **Tone and Register**
       - Formality level
       - Emotional undertones
    
    5. **Figurative Language and Rhetorical Devices**
       - Types used
       - Examples and their effects
    
    6. **Stylistic Nuances**
       - Unique elements
       - Their contribution to the voice
    
    Present the style guide in a clear and organized format for easy reference in future content generation tasks.
    
    ---
    **General Analysis:**
    {analyses['general_analysis']}
    
    **Lexical Analysis:**
    {analyses['lexical_analysis']}
    
    **Syntactic Analysis:**
    {analyses['syntactic_analysis']}
    
    **Structural Analysis:**
    {analyses['structural_analysis']}
    
    **Tone and Register Analysis:**
    {analyses['tone_and_register']}
    
    **Figurative Language and Rhetorical Devices:**
    {analyses['figurative_language']}
    
    **Stylistic Nuances:**
    {analyses['stylistic_nuances']}
    ---
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2)
    style_guide = response.choices[0].message.content.strip()
    return style_guide

def create_article_outline(style_guide: str, bullet_points: List[str]) -> str:
    """
    Creates a detailed outline for the new article based on the style guide and bullet points.
    """
    bullet_points_text = "\n- ".join(bullet_points)

    prompt = f"""
    Based on the following main bullet points and adhering to the style guide, create a detailed outline for a high-end magazine article. The outline should include:
    
    1. **Introduction**: Engaging opening to hook the reader.
    2. **Main Sections**: At least three main sections, each with a clear heading and sub-points.
    3. **Conclusion**: Thoughtful closing that reinforces the article's message.
    
    Main Bullet Points:
    - {bullet_points_text}
    
    ---
    **Style Guide:**
    {style_guide}
    ---
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2)
    outline = response.choices[0].message.content.strip()
    return outline

def generate_article(style_guide: str, outline: str) -> str:
    """
    Generates the full article based on the style guide and article outline.
    """
    prompt = f"""
    Using the detailed outline below and adhering to the provided style guide, write the full high-end magazine article. Ensure that each section is elaborated with rich descriptions, insightful analysis, and engaging narratives. Maintain consistency in tone and style throughout the article.
    
    ### **Article Outline**
    
    {outline}
    
    ---
    **Style Guide:**
    {style_guide}
    ---
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2,
    max_tokens=2000)
    article = response.choices[0].message.content.strip()
    return article

def validate_coherence(article: str, style_guide: str) -> str:
    """
    Reviews the article for coherence, logical flow, and adherence to the style guide.
    """
    prompt = f"""
    Review the following article for coherence, logical flow, and adherence to the high-end magazine style as outlined in the style guide. Identify any sections that lack clarity, have inconsistent tone, or break the logical progression. Provide feedback and suggest revisions where necessary.
    
    ### **Generated Article**
    
    {article}
    
    ###
    **Style Guide:**
    
    {style_guide}
    ---
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an experienced editor."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2)
    feedback = response.choices[0].message.content.strip()
    return feedback

def refine_article(feedback: str, article: str) -> str:
    """
    Incorporates feedback to refine and polish the article.
    """
    prompt = f"""
    Incorporate the feedback provided in the previous stage to refine the article. Enhance the language, correct any inconsistencies, and ensure the final version meets the high standards of a premier magazine publication.
    
    **Feedback Summary:**
    {feedback}
    
    ### **Original Article**
    
    {article}
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a meticulous editor."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2,
    max_tokens=2000)
    revised_article = response.choices[0].message.content.strip()
    return revised_article

# ================================
# Define the Workflow Execution Function
# ================================

def generate_high_end_article(sample_text: str, bullet_points: List[str]) -> str:
    """
    Generates a high-end magazine article based on the sample text and bullet points.
    
    Args:
        sample_text (str): The reference article to extract style from.
        bullet_points (List[str]): The main content bullet points for the new article.
    
    Returns:
        str: The final polished article.
    """
    print("Stage 1: Confirm Receipt of Sample Text")
    confirmation = confirm_receipt(sample_text)
    print(f"Confirmation: {confirmation}\n")

    print("Stage 2: General Analysis")
    general = general_analysis(sample_text)
    print(f"General Analysis: {general}\n")

    print("Stage 3: Lexical Analysis")
    lexical = lexical_analysis(sample_text)
    print(f"Lexical Analysis: {lexical}\n")

    print("Stage 4: Syntactic Analysis")
    syntactic = syntactic_analysis(sample_text)
    print(f"Syntactic Analysis: {syntactic}\n")

    print("Stage 5: Structural Analysis")
    structural = structural_analysis(sample_text)
    print(f"Structural Analysis: {structural}\n")

    print("Stage 6: Tone and Register Analysis")
    tone = tone_and_register_analysis(sample_text)
    print(f"Tone and Register Analysis: {tone}\n")

    print("Stage 7: Figurative Language and Rhetorical Devices Analysis")
    figurative = figurative_language_analysis(sample_text)
    print(f"Figurative Language Analysis: {figurative}\n")

    print("Stage 8: Stylistic Nuances Analysis")
    nuances = stylistic_nuances_analysis(sample_text)
    print(f"Stylistic Nuances Analysis: {nuances}\n")

    print("Stage 9: Compile Style Guide")
    analyses = {
        "general_analysis": general,
        "lexical_analysis": lexical,
        "syntactic_analysis": syntactic,
        "structural_analysis": structural,
        "tone_and_register": tone,
        "figurative_language": figurative,
        "stylistic_nuances": nuances
    }
    style_guide = compile_style_guide(analyses)
    print(f"Style Guide:\n{style_guide}\n")

    print("Stage 10: Create Article Outline")
    outline = create_article_outline(style_guide, bullet_points)
    print(f"Article Outline:\n{outline}\n")

    print("Stage 11: Generate Article")
    article = generate_article(style_guide, outline)
    print(f"Generated Article:\n{article}\n")

    print("Stage 12: Validate Coherence")
    feedback = validate_coherence(article, style_guide)
    print(f"Coherence Feedback:\n{feedback}\n")

    print("Stage 13: Refine Article")
    refined_article = refine_article(feedback, article)
    print(f"Final Refined Article:\n{refined_article}\n")

    return refined_article

# ================================
# Example Usage
# ================================

if __name__ == "__main__":
    # Example sample article (Replace with your actual sample)
    sample_article = """
    # Najciekawsze wystawy w Polsce i na świecie, które warto zobaczyć w 2025 roku
    ### Cukierkowe malarstwo Wayne’a Thiebauda, szydełkowe rzeźby Ewy Pachuckiej i digitalowe instalacje Alice Bucknell to tylko niektóre z fascynujących dzieł sztuki, jakie zobaczymy na wystawach w 2025 roku. Wybraliśmy dziesięć muzealnych ekspozycji, obok których nie można przejść obojętnie.
    ## Wszystkie twarze Jawlensky’ego
    Wystawy twórczości Alexeja von Jawlensky’ego zdarzają się nadzwyczaj rzadko. Retrospektywa tego wybitnego ekspresjonisty, która otworzy się pod koniec stycznia w duńskim Louisiana Museum, jest więc wydarzeniem godnym uwagi. Zaprezentowanych na niej ponad 60 dzieł pozwoli prześledzić artystyczną drogę rosyjsko-niemieckiego malarza: jego bliskie kontakty z Wassilym Kandinskim i Gabriele Münter, członkostwo w słynnej grupie Der Blaue Reiter, a także eksperymenty z fowizmem, ekspresjonizmem i abstrakcjonizmem. Ekspozycja w szczególny sposób skoncentruje się na ludzkiej twarzy, którą Jawlensky malował wciąż na nowo, niemal obsesyjnie. Stanowiła dla niego motyw, w którym zawierała się istota egzystencji i duchowości. Z czasem jej forma stała się prawie całkowicie abstrakcyjna, osiągając kulminację w ostatniej malarskiej serii pt. „Medytacje”, powstałej w czasie, gdy artysta zmagał się z chorobą uniemożliwiającą mu dalsze tworzenie. Wystawa ukaże Jawlensky’ego jako niestrudzonego poszukiwacza form, mającego swój udział w ewolucji sztuki nowoczesnej.

    „Alexej von Jawlensky”, Louisiana Museum of Modern Art, Humlebæk, 30 stycznia – 1 czerwca 2025 roku
    ## Zjednoczone Stany Fotografii
    Rijksmuseum w Amsterdamie zaprezentuje pierwszą w Europie kompleksową wystawę fotografii amerykańskiej. Ekspozycja obejmie ponad 200 prac, tworzonych od końca XIX wieku do dziś. Ukaże bogatą historię fotograficznego medium w USA – jego obecność w sztuce, prasie, reklamie i życiu codziennym. Pracom znanych artystów, takich jak Paul Strand, Irving Penn, Robert Frank, Diane Arbus, Ming Smith czy Nan Goldin, towarzyszyć będą zdjęcia wykonane przez amatorów oraz anonimowych twórców. Wspólnie ujawnią one repertuar motywów i tematów, za pośrednictwem których poszczególni artyści dają wyraz swojej amerykańskiej tożsamości i społecznych doświadczeń. Ponadto obszerny przegląd fotograficznych praktyk zostanie dopełniony pokazem serii „Painting the Town” autorstwa Carrie Mae Weems, laureatki Hasselblad Award, przyznawanej wybitnym osobistościom sztuki fotograficznej.

    „American Photography”, Rijksmuseum, Amsterdam, 7 lutego – 9 czerwca 2025 roku
    ## Danse macabre
    Czy kluczem do zrozumienia sztuki XIX i XX wieku mogą być dzieła gotyckie i renesansowe? Dlaczego wybierając makabryczne motywy, artyści nowoczesności inspirowali się dziełami sprzed kilku stuleci? Takie pytania zadaje monumentalna wystawa „Gothic Modern: From Darkness to Light” w norweskim Nasjonalmuseet. Według organizatorów tytułowy gotyk – kojarzony z mrokiem, grozą i enigmatycznością – daje asumpt do refleksji nad wiarą i zwątpieniem, mistyką i materialnością, przemijaniem i wiecznością. Prezentowane dzieła podejmują temat życia i śmierci, nie stroniąc od drastycznego obrazowania czy czarnego humoru. Na ekspozycji znajdzie się 250 prac autorstwa m.in. Käthe Kollwitz, Edvarda Muncha, Vincenta van Gogha, Arnolda Böcklina, Helene Schjerfbeck i Ernsta Ludwiga Kirchnera, które zostaną zestawione z twórczością starych mistrzów, takich jak Hans Holbein, Lucas van Leyden, Albrecht Dürer czy Lucas Cranach Starszy.

    „Gothic Modern: From Darkness to Light”, Nasjonalmuseet, Oslo, 28 lutego – 15 czerwca 2025 roku
    ## Awangarda z widokiem na morze
    Nikt nie spodziewał się, że St Ives – ciche rybackie miasteczko na zachodnim krańcu Kornwalii – stanie się oazą dla kilku pokoleń awangardowych artystów. Malowniczości nadmorskiego krajobrazu ulegli m.in. Ben Nicholson, Barbara Hepworth, Christopher Wood, Wilhelmina Barns-Graham, Naum Gabo i Patrick Heron. Choć w historii sztuki znani są jako „St Ives School”, to jednak nie tworzyli formalnej grupy, której działalność określona byłaby jednolitym programem. Mimo to lokalny pejzaż stanowił dla nich punkt wyjścia do definiowania na nowo założeń abstrakcjonizmu, zasad kompozycji i charakteru medium artystycznego. Wystawa „St Ives, i gdzie indziej” w Muzeum Sztuki w Łodzi przybliży historię tej artystycznej kolonii. Prace Brytyjczyków zostaną zaprezentowane w relacji z dziełami rodzimych twórców, m.in. Saszy Blondera, Katarzyny Kobro i Władysława Strzemińskiego. Jak mówi kurator łódzkiej ekspozycji Paweł Polit: – Łącznik między obydwoma środowiskami stanowi twórczość Piotra Potworowskiego, którego pobyt w Polsce w latach 1958–1962 był okazją do zastosowania idei wypracowanych przez niego w Wielkiej Brytanii, w artystycznym dialogu ze środowiskiem brytyjskim, w kontekście polskim.
    
    „St Ives, i gdzie indziej”, Muzeum Sztuki w Łodzi, 5 marca – 8 czerwca 2025 roku
    ## Apetyczny pejzaż Ameryki
    Piętrowe torty, ciastka z kremem, karmelizowane jabłka na patyku, lizaki, gumy balonowe, a także hamburgery, pieczone kurczaki i butelki oranżady – to bohaterowie obrazów Wayne’a Thiebauda. Jego martwe natury z barowych witryn, delikatesowych regałów czy półek w butikach przedstawiają realia dobrobytu w powojennych Stanach Zjednoczonych. Ukazują zarazem błogostan i nienasycenie amerykańskiego społeczeństwa. Obrazy Thiebauda stanowią krytykę nadmiaru i konsumpcji, jednak ukrytą pod lukrem sentymentalizmu i nostalgii za prostszymi czasami. Jego malarstwo jest równocześnie karykaturalne i przejmująco poważne, tandetne i kunsztowne, banalnie proste i semantycznie zawiłe. Twórczości jednego z najoryginalniejszych malarzy Zachodniego Wybrzeża USA poświęcona zostanie monograficzna wystawa w Fine Arts Museums w San Francisco. Znajdzie się na niej blisko 60 prac z lat 1957-2020. Obok słynnych pop-artowych martwych natur zaprezentowane zostaną również (mniej znane, lecz równie fascynujące pod względem formalnym) portrety, reinterpretacje dzieł dawnych mistrzów, a także pejzaże gór oraz kalifornijskich miast.

    „Wayne Thiebaud: Art Comes from Art”, Legion of Honor – Fine Arts Museums of San Francisco, 22 marca – 17 sierpnia 2025 roku
    ## Polskie artystki w pracowni Bourdelle’a
    W historii sztuki Émile Antoine Bourdelle znany jest nie tylko jako pomocnik słynnego Auguste’a Rodina oraz ceniony rzeźbiarz wczesnego modernizmu – autor „Herkulesa napinającego łuk”, pomnika Adama Mickiewicza w Paryżu czy reliefów na fasadzie Théâtre des Champs-Elysées. Zasłynął też jako wybitny pedagog w legendarnej Académie de la Grande Chaumière na Montparnassie. W pracowni francuskiego mistrza szkolili się Alberto Giacometti, Aristide Maillol i Wiera Muchina, a także grono Polek i Polaków. Majowa wystawa w warszawskiej Królikarni przybliży twórcze losy polskich artystek, które doskonaliły swój warsztat pod okiem Bourdelle’a. Zobaczymy prace m.in. Marii Lednickiej-Szczytt, Luny Drexler, Olgi Niewskiej, Miki Mickun i Kazimiery Małaczyńskiej-Pajzderskiej. Ekspozycja ujawni wpływ nauk mentora na jego uczennice, jak również ich własne poszukiwania indywidualnego języka ekspresji.

    „Kierunek Paryż. Polskie artystki z pracowni Bourdelle’a”, Królikarnia – Muzeum Rzeźby im. Xawerego Dunikowskiego, Warszawa, 9 maja – 26 października 2025 roku
    ## Radykalistki
    Na wystawie „Radical!” w wiedeńskim Belwederze zgromadzone zostaną dzieła ponad 60 wybitnych artystek sztuki modernistycznej, m.in. Claude Cahun, Soni Delaunay, Alexandry Exter, Hannah Höch, Katarzyny Kobro, Sophie Taeuber-Arp i Fahrelnissy Zeid. Choć różniły je generacja, pochodzenie i styl, to jednak łączyła wspólna misja kreowania nowych języków wizualnych oraz form reprezentacji, mających odzwierciedlać współczesny im świat. Ich prace w sposób poruszający dokumentują społeczne przeobrażenia pierwszej połowy XX wieku, a także problemy politycznych zawirowań i technologicznych przemian. Wystawa eksploruje indywidualny i zróżnicowany charakter działań malarek, rzeźbiarek i fotografek: od abstrakcji do figuratywności, od krytyki po aktywizm. Jednocześnie rozbija ideę linearnego następowania po sobie ruchów awangardowych i uwalnia twórczość artystek z historycznych mechanizmów, które przyczyniły się do ich zapomnienia. Ekspozycja rzuca wyzwanie męskocentrycznej historii, która marginalizowała lub wręcz wykluczała kobiety z kanonu sztuki nowoczesnej.

    „Radical! Women Artists and Modernism 1910-1950”, Belvedere, Wiedeń, 18 czerwca – 12 października 2025 roku
    ## Światy wyszydełkowane
    Ewa Pachucka znana jest lepiej w Australii (gdzie spędziła blisko 30 lat życia) niż w Polsce. Jej prace znajdują się w kolekcji National Gallery of Australia w Canberze czy National Gallery of Victoria w Melbourne. Planowana na jesień retrospektywa w warszawskiej Zachęcie stanowi więc okazję, by na nowo odkryć twórczość jednej z najciekawszych reprezentantek Polskiej Szkoły Tkaniny. Kreowane przez nią rzeźbiarskie szydełkowe struktury z włókna konopnego, juty czy sizalu w oryginalny sposób odnoszą się do ludzkiej cielesności, a także związku człowieka z naturą. Prócz rzeźb i instalacji wykonanych w medium tkaniny na ekspozycji zaprezentowane zostaną również monotypie i reliefy. Jak zauważa kurator wystawy Michał Jachuła: – Porzucenie dekoracyjności na rzecz eksponowania surowości formy i materii oraz bardzo wczesne uprzestrzennienie prac sytuują sztukę Pachuckiej jako wyjątkowo nowatorską na międzynarodowej scenie artystycznej XX wieku.

    „Ewa Pachucka. Retrospektywa”, Zachęta – Narodowa Galeria Sztuki, Warszawa, od października 2025 do stycznia 2026 roku (dokładny termin zostanie ogłoszony wkrótce)
    """

    # Example bullet points for the new article
    bullet_points = [
        "Początki cyfrowej sztuki",
        "Rewolucja interaktywności i multimedia",
        "Sztuka generatywna i sztuczna inteligencja."
        "NFT i cyfrowa własność w sztuce"
        "Kontrowersje i przyszłość technologicznej sztuki"
    ]

    # Generate the article
    final_article = generate_high_end_article(sample_article, bullet_points)

    # Print the final article
    print("\n=== Final Article ===\n")
    print(final_article)