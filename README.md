# NLPBridge 0.2.0 — Python + Rust

Repozytorium: https://github.com/autogrammar/nlpbridge. Dystrybucja `nlpbridge`
i API `from nlpbridge import select_operation` pozostają kompatybilne.
Polecenia `nlpbridge` i `nlbridge` uruchamiają ten sam CLI.

Uruchamialny szkielet NL → plan JSON → DSL lub jawnie podłączone API.
Python odpowiada za integrację z aplikacją i modelami. Rust przyspiesza skanowanie
indeksu wektorowego przez bezpośrednie FFI. Projekt nie zawiera słowników intencji
PL/EN, reguł kolejności słów ani parsera języka naturalnego opartego na regexach.
Interpretację języka wykonuje podłączony model.

Wersja 0.2.0 dodaje krótsze karty wyboru operacji, cache `f32le/1`, kontrolę
integralności lokalnych modeli oraz dwa backendy embeddingów ONNX. W pomiarze
E5 oba adaptery osiągnęły medianę około 6 ms, bez wykazanej trwałej przewagi Rusta.
Rust ONNX pozostaje opcjonalny. Dokładne wyniki i zakres pomiaru: `reports/REPORT.md`.

## Szybki start

Wymagania: Python 3.11+, opcjonalnie stabilny toolchain Rust z Cargo.
Polecenia uruchamiaj w katalogu projektu.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python tools/build_native.py
nlbridge doctor
python examples/in_process.py
```

Na Windows aktywuj `.venv\Scripts\activate` i zbuduj bibliotekę tym samym skryptem.
`NLBRIDGE_CARGO` może wskazywać bezwzględną ścieżkę do Cargo. Bez kompilacji
`backend = "auto"` korzysta z Pythona. Po zbudowaniu biblioteki uruchom proces ponownie.
Dołączone binaria są wyłącznie dla sprawdzonego środowiska Linux x86_64;
własna kompilacja jest właściwą ścieżką na innych systemach lub starszym glibc.
Przy instalacji nieedytowalnej (`pip install .`) wskaż zbudowaną bibliotekę
zmienną `NLBRIDGE_NATIVE=/pelna/sciezka/libnlbridge_core.so` (na Windows/macOS
użyj odpowiedniego rozszerzenia). Pakiet Pythona pozostaje niezależny od ABI CPythona.

Przykład działający bez LLM, z jawnym URI i argumentami:

```bash
nlbridge compile 'proc://demo/text/word-count/v1' \
  --args '{"text":"Zażółć gęślą jaźń"}'
```

Wynik zawiera `status`, plan, DSL i czasy poszczególnych etapów. Samo `compile`
nie wykonuje operacji. Przykładowy DSL:

```text
# nlbridge/dsl-v1
s1 = call("proc://demo/text/word-count/v1", {"text":"Zażółć gęślą jaźń"});
```

## Prawdziwe zapytania NL

Przykładowa konfiguracja Ollama używa istniejącego tagu `qwen3.5:2b`.
To punkt startowy do pomiarów, nie zwycięzca benchmarku językowego tego projektu.

```bash
ollama pull qwen3.5:2b
# Ollama musi działać pod adresem ustawionym w konfiguracji.
nlbridge --config examples/ollama.toml compile \
  'Konwertuj logo.svg do logo.png i zachowaj oryginał.'
```

Alternatywa: uruchom własny `llama-server` z modelem GGUF i aliasem `local`
na porcie 8081, a następnie użyj `examples/llamacpp.toml`. Wpis `revision`
zmieniaj przy zmianie wag, kwantyzacji lub serwera. Dla HTTP jest to deklarowana
przez operatora tożsamość wdrożenia, a nie dowód integralności zdalnego modelu.

Adapter wysyła JSON Schema przez Ollama `format` albo przez
`response_format.json_schema` serwera zgodnego z OpenAI. Obsługiwany podzbiór
schematu zależy od serwera. Błąd serwera nie powoduje przejścia na swobodny tekst.
Walidacja po generowaniu jest zawsze wykonywana. Protokoły obu adapterów zostały
sprawdzone z lokalnym serwerem testowym. Paczka zawiera pomiary rzeczywistych
embeddingów E5; nie zawiera pomiaru inferencji LLM ani deklaracji jakości dla
wszystkich języków europejskich.

Cztery przykładowe operacje mieszczą się w `top_k = 8`, dlatego konfiguracja
startowa nie potrzebuje embeddingów. Wszystkie dozwolone kontrakty trafiają do
modelu. Przy większym katalogu włącz model embeddingowy obsługujący wymagane języki.

Przykładowa sekcja lokalnych embeddingów, zastępująca `[embedding]`:

```toml
[embedding]
provider = "sentence_transformers"
model = "intfloat/multilingual-e5-small"
revision = "WSTAW_PELNY_HASH_COMMITU_Z_REPOZYTORIUM_MODELU"
backend = "onnx"
query_prefix = "query: "
document_prefix = "passage: "
```

Zainstaluj `python -m pip install -e '.[models]'`. Użyj rzeczywistego commitu,
sprawdź wymagania wybranego modelu i jego tokenizera. ONNX może zostać wyeksportowany
przy pierwszym ładowaniu. Długie opisy podlegają limitowi modelu embeddingowego;
utrzymuj kontrakty krótkie. Adaptery HTTP embeddingów obsługują także Ollama
`api/embed` oraz zgodne endpointy `/v1/embeddings`; wymagają `model`, `revision`,
`base_url` i `dim`. Prefiksy zapytania i dokumentu są konfigurowalne.

## Lokalne ONNX: sprawdzona ścieżka w 0.2.0

```bash
python -m pip install -e '.[onnx]'
python tools/prepare_e5.py
nlbridge --config examples/onnx.toml index
nlbridge --config examples/onnx.toml status
# Opcjonalnie: natywna tokenizacja, inferencja i pooling przez C ABI.
python tools/build_native.py --onnx
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/benchmark_onnx.py --trials 100
```

Skrypt pobiera około 136 MB plików z przypiętego commitu
`614241f622f53c4eeff9890bdc4f31cfecc418b3` repozytorium
`intfloat/multilingual-e5-small`: wariant `model_qint8_avx512_vnni.onnx`, tokenizer
i kartę modelu. To wariant zmierzony na CPU x86_64 tej sesji; sprawdź go na własnym
sprzęcie. Wagi nie są w ZIP. Pobieranie następuje tylko po jawnym uruchomieniu skryptu.

`examples/onnx.toml` domyślnie używa `provider = "onnx"`. Zmień na `"rust_onnx"`,
aby korzystać z opcjonalnej biblioteki. Obie ścieżki używają tego samego pliku
ONNX Runtime 1.24.4, ustawień wątków i bundla. Pythonowy adapter ONNX również
wykonuje inferencję w kodzie natywnym. W przykładzie `[model]` pozostaje `none`;
do NL → plan dodaj konfigurację LLM z `ollama.toml` lub `llamacpp.toml`.

Bundle `manifest.json` wiąże SHA-256 grafu, tokenizera i wszystkich zewnętrznych
plików wag z parametrami przetwarzania: prefiksami, paddingiem, limitem tokenów,
poolingiem i normalizacją. Hashowanie i sprawdzenie grafu odbywają się przy
ładowaniu. Deskryptor cache uwzględnia też bibliotekę runtime, implementację
i ustawienia; przeniesienie całego bundla do innego katalogu nie zmienia jego
tożsamości. Traktuj pliki załadowanego bundla jako niezmienne; po zmianie uruchom
nowy embedder i `store.sync()`.

Własny wyeksportowany model przygotujesz przez
`nlbridge.model_bundle.create_bundle(directory, ...)`. Podaj jego rzeczywiste
`dim`, `pad_id`, `max_length`, prefiksy, `output_name` i `pooling` (`mean`, `cls`
lub `none`). Obsługiwane wejścia to rank-2 int32/int64 `input_ids`,
`attention_mask` i opcjonalne `token_type_ids`; output to float32. To adapter
dla tej rodziny grafów, nie dla dowolnego modelu ONNX. Dłuższe wejście daje błąd
bez obcinania tekstu. Długie kontrakty skróć jawnie w `selection_description`
lub użyj odpowiednio pojemnego modelu. Zwykły adapter Sentence Transformers
zachowuje odrębne ustawienia truncation wybranego modelu.

Natywna biblioteka jest ładowana z `python/nlbridge/_native`. Przy instalacji
nieedytowalnej ustaw `embedding.native_path` na zbudowany plik. Opcjonalne
`embedding.runtime_library` wskazuje własną zgodną bibliotekę ONNX Runtime
(API 24); w przeciwnym razie używana jest biblioteka z pakietu `onnxruntime`.
Proces może załadować tylko jedną wersję tej biblioteki. Nagłówek C ABI:
`rust/onnx/include/nlbridge_onnx.h`. Bezpośredni klient C sam weryfikuje bundle
przed otwarciem sesji. Domyślny build rdzenia nadal nie zawiera ONNX.

## Krótszy wybór operacji

`runtime.selection_view = "compact"` wysyła URI, opis, efekty i skrócone schematy.
Zachowuje m.in. `const`, `enum`, `pattern`, pola wymagane i ograniczenia kierunku.
Pełne schematy pozostają w fazie argumentów i walidacji. Opcjonalne pole kontraktu
`selection_description` jest jawnym opisem autora; nie stosujemy automatycznego
obcinania opisów. `selection_view = "full"` umożliwia porównanie obu wariantów.

Na czterech przykładowych operacjach karty zmniejszyły payload wyboru z 2119 do
1834 bajtów (13,45%). To rozmiar danych operacji, nie liczba tokenów ani przyspieszenie
LLM. Wynik `compile()` raportuje `metrics.prompt_bytes` i
`prompt_bytes_by_phase`; metryka obejmuje również ewentualną naprawę odpowiedzi.

## Podział odpowiedzialności

| Warstwa | Implementacja | Zachowanie |
|---|---|---|
| Katalog | Python + SQLite | Import JSON/JSONL/SQLite, aktualizacja różnicowa, cache embeddingów według treści i wersji modelu |
| Retrieval | Rust przez `ctypes.CDLL` | Dokładny cosine top-k, maskowanie dozwolonych operacji przed rankingiem, stabilne remisy |
| Retrieval leksykalny | Python | Tokenizacja Unicode, exact-match URI, fuzja RRF; bez list słów charakterystycznych dla języków |
| Planowanie | Python + podłączony LLM | Najpierw wybór operacji, następnie argumenty według małego schematu wybranej operacji |
| Semantyka planu | Python | JSON Schema, effects/backend/status/URI, rewizja katalogu, digest, zgodność referencji |
| Rdzeń planu | Rust lub Python | Struktura grafu, URI i bezpieczne kodowanie DSL |
| Integracja | Python | Funkcje aplikacji, adaptery API, CLI oraz endpoint HTTP |

Wektory są kopiowane do Rusta raz przy budowie indeksu. Każde zapytanie przekazuje
tylko wektor zapytania, maskę i limit. Między Pythonem a Rustem nie ma HTTP ani
procesu uruchamianego dla każdego zapytania. Indeks jest niemutowalny, a wywołanie
`ctypes.CDLL` zwalnia GIL na czas pracy funkcji natywnej.

`catalog.backend`: `auto`, `rust`, `python` lub `numpy`. `auto` wybiera natywny
indeks, jeśli biblioteka jest dostępna. Dla krótkiego DSL `auto` pozostawia
serializację w Pythonie, zgodnie z pomiarem narzutu FFI. `rust` wymusza także
natywny renderer. Opcjonalny backend NumPy instalujesz przez `.[numpy]`.

Wyszukiwanie jest dokładne, O(N × D), i nie używa ANN. Dla bardzo dużych katalogów
wymień `Snapshot.search`/`VectorIndex` na indeks ANN po pomiarze recall i p95.
Benchmark rdzenia nie uwzględnia pełnego RRF, tokenizacji ani inferencji embeddingów.

## Kontrakty, odmowa i dataflow

Operacja deklaruje URI, opis, input/output JSON Schema, effects i backend.
Nowy projekt dodaje kontrakty oraz powiązania wykonawcze, bez dopisywania reguł
językowych. W `examples/catalog.json` kierunek konwersji jest wyrażony kontraktem
i ograniczeniami rozszerzeń plików. To reguły domenowe; nie rozpoznawanie gramatyki NL.

Statusy odpowiedzi:

- `ready`: kompletny plan, który przeszedł walidację;
- `clarify`: pytanie i brakujące dane, bez planu do wykonania;
- `unsupported`: brak odpowiedniej operacji w dostępnych kandydatach/polityce.

Błędy transportu lub dwukrotnie błędny wynik modelu są osobnym `ModelError`.
Pusta lista kandydatów nie dowodzi, że operacja nie istnieje w dużym katalogu.
Pierwszy błąd struktury/semantyki powoduje jedną próbę naprawy z treścią błędu.
Wybór operacji wymaga jednego wywołania LLM, wiązanie N kroków wymaga kolejnych N.
Temperatura 0 nie stanowi gwarancji deterministycznej lub poprawnej interpretacji.

IR zawiera `catalog_revision` oraz `digest` każdej operacji. Referencja argumentu:

```json
{"$ref":{"step":"s1","pointer":"/text"}}
```

Referencje dotyczą argumentów najwyższego poziomu i gwarantowanych, wymaganych
pól obiektowych poprzednich wyników. Nie ma odwołań do przyszłych kroków, cykli,
pętli, warunków ani odwołań do elementów tablic. Dowodzenie zgodności typów jest
konserwatywnym podzbiorem JSON Schema; odrzucony przypadek może wymagać adaptera.
Schematy z referencjami JSON Schema rozwiąż przed importem. `format` jest adnotacją
Draft 2020-12, nie pełną walidacją dat/URI przez FormatChecker. Po rozwiązaniu
referencji ponownie sprawdzane są argumenty i rzeczywisty output każdego kroku.

## Włączenie do aplikacji / NL → API

```python
from nlbridge.config import open_runtime
from nlbridge.executor import execute

runtime, stats = open_runtime("examples/ollama.toml")
try:
    result = runtime.compile("Policz słowa w tekście 'ala ma kota'.")
    if result["status"] == "ready":
        bindings = {
            "proc://demo/text/word-count/v1":
                lambda text: {"count": len(text.split())},
        }
        outputs = execute(
            result["plan"], runtime.store.snapshot, bindings, runtime.policy
        )
finally:
    runtime.store.close()
```

W `bindings` umieść własną funkcję wywołującą API ze stałym, zaufanym endpointem
i poświadczeniami zarządzanymi przez aplikację. Model wybiera URI kontraktu;
mapowanie URI → callable/API pozostaje w kodzie aplikacji. Przykładowe operacje
konwersji obrazów są kontraktami do podłączenia do własnego backendu. Paczka nie
implementuje rasteryzacji ani wektoryzacji obrazów. Przykład liczenia słów wykonuje
mechaniczne `text.split()`; nie jest biblioteką segmentacji językowej.

Do własnego DSL podłącz renderer przyjmujący zwalidowany IR. Do wykonania używaj
IR i aktualnego katalogu, nie `eval()` wygenerowanego tekstu DSL. Sam renderer
natywny sprawdza strukturę, nie zastępuje walidacji kontraktów i uprawnień.

Możesz wskazać w `catalog.path` istniejącą bazę z tabelą `operations(record)`.
Importer otwiera ją tylko do odczytu i pobiera `status = declared`. Kandydaci ze
skanowania kodu nie są kontraktami. `catalog.database` musi wskazywać oddzielną bazę
cache NLBridge. Import nie uruchamia skanowanego kodu i nie przenosi pola `exec`
do planu. Jeśli aplikacja używa `verify_record()`, zachowaj weryfikację źródeł przed
wykonaniem własnego adaptera. Digest z importu nie weryfikuje sam plików na dysku.

## HTTP i granica wykonania

```bash
nlbridge --config examples/ollama.toml serve --port 8080 --workers 4
curl http://127.0.0.1:8080/v1/compile \
  -H 'Content-Type: application/json' \
  --data '{"text":"Konwertuj logo.svg do logo.png i zachowaj oryginał."}'
```

`GET /health`, `GET /ready` i `POST /v1/compile`; nie ma zdalnego endpointu uruchamiającego kod.
Serwer domyślnie słucha na localhost. Do wystawienia poza host użyj uwierzytelnionego
reverse proxy. Polityka pochodzi z konfiguracji serwera; żądanie klienta nie może
jej zmienić. Serwer ma ograniczoną liczbę workerów, limit 64 KiB body i zwraca 503
przy pełnym zajęciu workerów. Połączenia do modeli są utrzymywane per worker;
odpowiedzi aplikacyjnego HTTP zamykają połączenie po odpowiedzi.

`/health` zwraca stan katalogu, generację indeksu, aktywne wektory, backend,
tożsamość modelu, ostatni sukces/błąd i `syncing`. `/ready` zwraca 503, gdy brak
opublikowanego indeksu albo jego embedder nie pasuje do konfiguracji. Gotowość
dotyczy lokalnego katalogu i kompilacji jawnego URI; `model_configured` mówi o
konfiguracji LLM. Endpoint nie odpytuje serwera modelu i nie potwierdza jego
dostępności. Po błędzie przebudowy poprzedni snapshot nadal obsługuje zapytania,
a diagnostyka pokazuje `degraded`.

Query, context, opisy i output modelu są danymi niezaufanymi. Walidacja schematu
ogranicza strukturę, ale nie usuwa ryzyka błędnej interpretacji/prompt injection
w obrębie dozwolonych operacji. Manifest i `declared` nie nadają uprawnień.
Host definiuje allowlisty, bindingi, walidację ścieżek/zasobów i uprawnienia użytkownika.
Przykładowa polityka dopuszcza `write` do planowania konwersji; to nie zgoda na
wykonanie operacji. `execute()` uruchamiaj dopiero po decyzji aplikacji.
Sekwencja efektów nie jest transakcją i nie ma automatycznego rollbacku.

Cache odpowiedzi uwzględnia query, context, argumenty, politykę, model, embedding,
rewizję katalogu i wersję promptu. Przechowuje wyłącznie poprawne plany, nigdy wyniki
wykonania. Cache jest w pamięci procesu, bounded LRU. Przy zmianie kontraktów wywołaj
`store.sync(records, embedder)`; publikacja nowego snapshotu następuje po sukcesie.
Stary plan zostanie odrzucony przez walidację aktualnego katalogu. Wagi modelu nie
są pobierane ponownie na każde zapytanie. Cache SQLite przechowuje wektory w jawnym
formacie little-endian float32 `f32le/1`. Stare wpisy bez identyfikacji formatu
oraz uszkodzone wektory są przeliczane. Nowy szablon kart również powoduje jednorazowe
przeliczenie indeksu z 0.1.0. Transakcja i publikacja snapshotu następują po sukcesie;
nieudana inferencja nie usuwa działającego indeksu. Cache może rosnąć przy kolejnych
wersjach modeli; stare wersje wymagają retencji ustalonej przez aplikację.

## Testy i pomiary

```bash
cargo test --locked
python -m unittest discover -s tests -v
NLBRIDGE_TEST_BACKEND=python python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/benchmark.py --trials 30
python tools/evaluate.py --config examples/ollama.toml --cases examples/eval.jsonl
# Pełny zestaw po instalacji .[onnx] i zbudowaniu opcjonalnej biblioteki:
python tools/verify.py --onnx
```

`reports/benchmark.json` zachowuje rzeczywiste pomiary niezmienionego algorytmu top-k
z wydania 0.1.0: p50, p95, sprzęt, backend i liczbę prób. Porównuje implementacje, nie wszystkie
możliwe optymalizacje NumPy/Rusta. `reports/verification.json` i logi określają
dokładnie, co uruchomiono. Zestaw testowy używa odpowiedzi kontrolowanych przy
sprawdzaniu kontraktów/protokołów; nie mierzy jakości rozumienia NL.

`reports/onnx-benchmark.json` obejmuje prawdziwe inferencje E5, porównanie wektorów
i rankingów Python/Rust, 100 rozgrzanych próbek na backend oraz ponowne użycie cache.
Mały test retrieval to 20 ręcznie napisanych zapytań PL/EN/DE/FR/ES na czterech
operacjach. Nie jest reprezentatywnym pomiarem jakości. Zbliżone operacje o odwrotnym
kierunku nadal wymagają rozstrzygnięcia przez plan i walidację.

`tools/evaluate.py` wykonuje rzeczywiste wywołania skonfigurowanego LLM, wyłącza
cache wyników i raportuje exact-plan accuracy oraz p50/p95 osobno dla języków.
Dziewięć przypadków PL/EN/DE/FR/ES to przykłady formatu, nie reprezentatywny benchmark.
Rozbuduj zestaw do własnych 50–200 przypadków na język: kierunek, negacja, zachowanie
źródła, brak danych, unsupported, injection i plany wielokrokowe. Pierwsze zapytanie
eval może zawierać koszt ładowania modelu. Porównuj również rozgrzane wdrożenia,
Q4/Q5/Q8, recall@k retrieval oraz p95 pod równoległym obciążeniem. Nie ma tu
obietnicy konkretnego p95 całego NL → DSL bez pomiarów modelu na docelowym sprzęcie.

Opcjonalny `Dockerfile` buduje bibliotekę natywną we własnym etapie i uruchamia
usługę jako użytkownik bez uprawnień roota. Docker i workflow GitHub Actions są
dostarczone jako konfiguracje; ich wykonanie wymaga odpowiednich środowisk.
Model uruchomiony na hoście wymaga adresu osiągalnego z kontenera. Przed wydaniem
produkcyjnym przypnij obrazy i zależności opcjonalnych modeli.

## Interfejs Rusta bez Pythona

`cargo build --release --locked` tworzy `target/release/nlbridge-core` oraz
bibliotekę `cdylib`. Nagłówek ABI: `rust/core/include/nlbridge.h`.
CLI Rusta to proces JSONL z komendami `load`, `search`, `render`, przykładowo:

```jsonl
{"command":"load","vectors":[[1,0],[0,1]]}
{"command":"search","query":[1,0],"mask":[0,1],"k":1}
```

Rdzeń Rusta nie zawiera inferencji LLM ani pełnego wykonawcy projektu. Cały potok
NL jest składany przez Pythona; natywny rdzeń pozostaje niezależny od modelu.

## Źródła interfejsów

- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama API chat](https://docs.ollama.com/api/chat)
- [Tagi Qwen 3.5 w Ollama](https://ollama.com/library/qwen3.5/tags)
- [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Sentence Transformers: ONNX](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
- [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
- [Rust FFI](https://doc.rust-lang.org/nomicon/ffi.html)
- [ort 2.0.0-rc.13](https://docs.rs/ort/2.0.0-rc.13/ort/)
- [ONNX Runtime: threads](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
- [Python ctypes](https://docs.python.org/3/library/ctypes.html)

Licencja kodu projektu: MIT. Wagi modeli i zależności mają własne licencje.
