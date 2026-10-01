# NLBridge 0.1.0 — wyniki weryfikacji

Pomiar i testy: 2026-10-01T07:55:15.479354+00:00. Platforma: Linux-6.18.44-x86_64-with-glibc2.39; CPU: AMD EPYC 9V74 80-Core Processor; Python 3.12.14. Rust 1.98.1.

## Testy

44 testy Pythona zaliczone w trybie natywnym oraz ponownie z wyłączoną biblioteką natywną. 7 testów Rusta zaliczonych. Przykład wykonania lokalnej funkcji również przeszedł. Szczegóły i kody wyjścia: `verification.json`; pełne logi obok.

Sprawdzone: zgodność top-k i wyników liczbowych Python/Rust/NumPy, filtrowanie przed rankingiem, kierunek określony kontraktem, typy referencji, odrzucenie cykli i odwołań do przyszłości, błędne wyjścia funkcji, polityka/cache, zmiana modelu embeddingowego, usuwanie rekordów, rollback nieudanej aktualizacji, HTTP i CLI. Kontrolowane odpowiedzi modelu testują mechanikę planowania i naprawy, nie rozumienie języka.

## Dokładne wyszukiwanie wektorowe

Wektory syntetyczne, 384 wymiary, top-8, 75% rekordów dopuszczonych maską, 30 prób po rozgrzaniu. Jeden wątek BLAS/OMP. Macierz już załadowana. Wynik obejmuje wywołanie z Pythona i transfer zapytania/maski przez FFI. Nie zawiera inferencji embeddingów, RRF, LLM ani komunikacji HTTP.

| Rekordy | Backend | p50 [ms] | p95 [ms] |
|---:|---|---:|---:|
| 1000 | python | 22.8704 | 27.1782 |
| 1000 | rust | 0.3533 | 0.4004 |
| 1000 | numpy | 0.3112 | 0.3988 |
| 10000 | python | 241.0923 | 282.6461 |
| 10000 | rust | 4.0028 | 5.6201 |
| 10000 | numpy | 5.3298 | 7.6963 |

Przy 10 000 rekordów Rust uzyskał w tym pomiarze około 60× krótszą medianę niż pętle Pythona. Przy 1 000 rekordów NumPy miał nieco niższą medianę niż Rust. Są to porównania konkretnych implementacji w tej paczce, bez gwarancji dla innego sprzętu, obciążenia lub alternatywnie zoptymalizowanej biblioteki. p95 z 30 prób jest orientacyjny.

## Kompilacja jawnego URI z argumentami

Krótki plan, działający proces, brak inferencji LLM i embeddingów, 100 prób. Wiersz Rust wymusza także natywną serializację; domyślne auto używa Pythona do serializacji małego DSL.

| Backend kompilacji | Cache wyniku | p50 [ms] | p95 [ms] |
|---|---|---:|---:|
| python | nie | 0.0494 | 0.1286 |
| python | tak | 0.0335 | 0.0597 |
| rust | nie | 0.0573 | 0.1395 |
| rust | tak | 0.0346 | 0.1009 |

## Zakres pozostawiony do sprawdzenia we wdrożeniu

Nie uruchamiano prawdziwego LLM ani lokalnego modelu ONNX. Nie zmierzono jakości wielojęzycznej, end-to-end p95 NL → DSL, recall@k embeddingów ani obciążenia współbieżnego. Skrypt `tools/evaluate.py` służy do pomiarów z rzeczywistym modelem i własnym zestawem przypadków. Wagi modeli nie są częścią paczki.

Docker, zdalny GitHub Actions oraz kompilacja macOS/Windows nie były uruchamiane w tym środowisku. Dołączone binaria są dla Linux x86_64; do innych systemów są źródła i skrypt budowania.

Instrukcja i ograniczenia semantyki planu: `../README.md`. Maszynowy wynik benchmarku: `benchmark.json`.
