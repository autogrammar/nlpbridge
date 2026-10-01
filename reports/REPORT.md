# NLBridge 0.2.0 — wdrożenie i weryfikacja

Wydanie rozszerza istniejący projekt Python + Rust. Zastosowano pomysły z przeglądu: krótkie karty operacji, identyfikację lokalnych modeli na podstawie plików, przenośny cache i diagnostykę. Dodano działający, opcjonalny moduł ONNX w Ruście oraz porównywalny adapter Python ONNX.

## Co zmieniono

- Karty wyboru operacji zachowują ograniczenia kontraktów; argumenty i dataflow nadal sprawdza pełny walidator. Pełny opis można zastąpić jawnym `selection_description`; tekst nie jest automatycznie obcinany.
- Lokalny bundle sprawdza SHA-256 grafu, tokenizera i zewnętrznych wag. Tożsamość obejmuje powiązania nazw plików z hashami, preprocessing i runtime. Przeniesienie katalogu nie zmienia tożsamości.
- Wektory mają format `f32le/1`. Migracja i odbudowa uszkodzonych wpisów zachowują dotychczasowy snapshot do sukcesu aktualizacji.
- `/health`, `/ready` i komenda `status` pokazują stan katalogu i indeksowania. Readiness nie jest testem dostępności zdalnego LLM.
- `rust/onnx` jest osobnym, opcjonalnym modułem C ABI. Rdzeń wyszukiwania nie ma zależności ONNX. Oba adaptery obsługują int32/int64, mean/CLS/gotowe wektory, maskowanie paddingu i normalizację. Przekroczenie limitu tokenów powoduje błąd.
- Metryki zawierają liczbę bajtów promptu per faza. Na przykładzie czterech operacji payload kart zmniejszył się z 2119 do 1834 bajtów (13,45%). Nie jest to pomiar liczby tokenów ani czasu LLM.

## Testy

Weryfikacja: 2026-10-01T08:50:23.503229+00:00. **59 testów Pythona zaliczonych** w konfiguracji rdzenia Rust, następnie tych samych 59 z rdzeniem Python. **9 testów Rusta zaliczonych**: 7 rdzenia i 2 modułu ONNX. Przykład lokalnego wykonania również przeszedł. Wszystkie kontrole `tools/verify.py --onnx` zakończyły się kodem 0.

Testy ONNX używają małych, generowanych grafów. Obejmują zgodność obu adapterów, int32/int64, padding, wywołania współbieżne, limity wejścia, błędny wymiar wyjścia, podmieniony tokenizer, brak hashy zewnętrznych wag i wyjście ścieżki poza bundle. Osobno sprawdzono naprawę cache oraz czytanie starego indeksu podczas nieudanej przebudowy. Testy LLM używają odpowiedzi kontrolowanych i nie mierzą rozumienia języka.

Pełne logi: `python-native.log`, `python-fallback.log`, `rust-tests.log`. Zbiorczy wynik: `verification.json`. Opcjonalne testy ONNX były uruchomione, nie pominięte. Tryb fallback wyłącza natywny rdzeń wyszukiwania; testy opcjonalnego adaptera ONNX nadal porównują oba jego backendy.

## Rzeczywisty model i opóźnienia

Pomiar: 2026-10-01T08:50:28.938364+00:00. CPU: AMD EPYC 9V74 80-Core Processor; platforma: Linux-6.18.44-x86_64-with-glibc2.39; Python 3.12.14. Oba adaptery: ONNX Runtime 1.24.4, tokenizers 0.22.2, CPU, 2 wątki intra-op, 1 inter-op, bez spinowania. BLAS/OMP: 1 wątek.

Model: `intfloat/multilingual-e5-small`, commit `614241f622f53c4eeff9890bdc4f31cfecc418b3`, plik `onnx/model_qint8_avx512_vnni.onnx`, 384 wymiary. Prefiksy `query: ` / `passage: `, mean pooling z maską, L2, limit 512 tokenów, pad ID 1. W raporcie JSON znajdują się hashe modelu, tokenizera, runtime i dołączonego modułu Rust.

100 rozgrzanych wywołań na backend; 20 krótkich zapytań PL/EN/DE/FR/ES, każdy powtórzony 5 razy. Kolejność losowana ze stałym seedem, backendy przeplatane. Czas obejmuje tokenizację, inferencję, pooling, normalizację i powrót listy wektorów do Pythona przez API lub FFI. Nie obejmuje LLM, HTTP, inicjalizacji ani całego NL → DSL.

| Adapter | p50 [ms] | p95 [ms] |
|---|---:|---:|
| Python ONNX | 6.0643 | 7.7399 |
| Rust ONNX | 5.9464 | 7.5741 |

Oba wyniki są bliskie 6 ms. W pomiarze wstępnym przewaga była po stronie Pythona, w końcowym po stronie Rusta. Te próby nie wykazują trwałej przewagi jednego adaptera; nie uzasadniają obietnicy przyspieszenia całego potoku. Python ONNX pozostaje domyślnym przykładem, Rust ONNX opcją integracyjną. Python również korzysta z natywnej inferencji ONNX Runtime.

Maksymalna różnica współrzędnych Python/Rust: 6.88e-09 dla zapytań, 4.86e-09 dla dokumentów; próg zgodności 1e-5. Wszystkie 20 rankingów było identycznych. Drugie indeksowanie w obu backendach wykonało **0 embeddingów** i **0 zapisów rekordów**.

Ładowanie wraz z weryfikacją plików: Python 1096.2 ms, Rust 1322.2 ms. To pojedyncze pomiary, z rozgrzanym cache systemu plików; nie są benchmarkiem zimnego startu.

## Mały test retrieval

Po cztery ręcznie napisane zapytania na język, cztery operacje w katalogu. W tabeli dense retrieval, bez RRF i LLM. To test działania modelu i zgodności adapterów, nie miarodajny benchmark wielojęzycznej jakości.

| Język | recall@1 | recall@2 |
|---|---:|---:|
| PL | 75% | 100% |
| EN | 75% | 100% |
| DE | 50% | 100% |
| FR | 50% | 100% |
| ES | 75% | 100% |

Wysokie recall@2 na czterech operacjach nie oznacza poprawnego wyboru kierunku konwersji. Wyniki wzmacniają potrzebę dwufazowego LLM i semantycznej walidacji argumentów. Surowe zapytania, rankingi, próbki czasowe i wyniki per język: `onnx-benchmark.json`.

## Migracja i uruchomienie

Kod konfiguracji i IR pozostają zgodne z 0.1.0. Pierwsze `store.sync()` przeliczy embeddingi ze względu na nowy szablon kart i oznaczenie formatu. `selection_view="full"` pozwala porównać dotychczasowy widok wyboru. Wagi pobiera osobny skrypt, nie import pakietu.

```bash
python -m pip install -e '.[onnx]'
python tools/prepare_e5.py
python tools/build_native.py --onnx
nlbridge --config examples/onnx.toml status
python tools/verify.py --onnx
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/benchmark_onnx.py --trials 100
```

Konfiguracja ONNX ma `[model] provider="none"`. Do interpretacji NL trzeba dodać endpoint LLM z przykładów Ollama/llama.cpp. Istniejąca ścieżka jawnego URI działa bez LLM. Przykłady konwersji obrazów nadal wymagają podłączenia własnych funkcji/API.

## Zakres dalszych pomiarów

Nie uruchamiano rzeczywistego LLM ani pomiaru end-to-end NL → DSL. Nie zmierzono p95 pod obciążeniem, reprezentatywnej jakości językowej ani Q4/Q5 planowania. `tools/evaluate.py` pozostaje narzędziem do oceny własnego LLM i zestawu przypadków. Docker, GitHub Actions, macOS i Windows nie zostały uruchomione. Dołączone biblioteki są dla Linux x86_64; ONNX Runtime oraz wagi pobierane osobno.

`benchmark.json` i `REPORT-v0.1.md` zachowują historyczny pomiar niezmienionego algorytmu top-k z wydania 0.1.0. Nie należy traktować tych czasów jako pomiarów inferencji lub czasu NL → DSL.
