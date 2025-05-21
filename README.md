# Optimierung von Text-zu-SQL-Modellen unter realistischen Hardware-Beschränkungen

Dieses Repository enthält Code, Ergebnisse und Dokumentationen im Rahmen einer Bachelorarbeit zur Untersuchung und Optimierung von Text-zu-SQL-Übersetzungsmodellen. Der Fokus lag auf der Evaluierung verschiedener Transformer-Architekturen (T5, DeepSeek) und Parameter-effizienter Feinabstimmungsmethoden (LoRA, QLoRA) auf dem Spider-Benchmark unter Berücksichtigung von Hardware-Limitationen (≤32 GB VRAM).

## Abstract
Diese Arbeit untersucht die Optimierung und den Vergleich verschiedener Architekturen von Transformer-Modellen – instruktionsoptimierte Encoder-Decoder-Modelle (T5-Familie) und code-vortrainierte Decoder-Only-Modelle (DeepSeek-Familie) – für die Text-zu-SQL-Übersetzung auf dem komplexen Spider-Benchmark. Der Fokus liegt auf der Maximierung der Übersetzungsgenauigkeit unter realistischen Hardware-Beschränkungen (≤32 GB VRAM) durch vollständiges Fine-Tuning sowie Parameter-effiziente Methoden wie LoRA und QLoRA.

Die Experimente zeigen, dass das vollständig feingetunte FLAN-T5-large Modell (770 Mio. Parameter) mit 62% Execution Accuracy (EX) und 61% Exact Match (EM) die höchste Leistung erzielt. Es übertrifft damit signifikant die getesteten DeepSeek-Modelle, einschließlich der DeepSeek-6.7B-LoRA-Variante (34% EX, 36% EM), was die Bedeutung des Instruktions-Tunings und der Encoder-Decoder-Architektur für diese Aufgabe unterstreicht. Die Studie demonstriert zudem, dass Parameter-effiziente Feinabstimmung (PEFT) erhebliche Ressourceneinsparungen ermöglicht: FLAN-T5-large-LoRA erreichte 46% EX bei nur 6 GB VRAM. Besonders hervorzuheben ist, dass QLoRA beim DeepSeek-1.3B Modell nicht nur den VRAM-Bedarf im Vergleich zum vollständigen Fine-Tuning um ca. 45% senkte, sondern auch die Genauigkeit steigerte (+7 pp EX). Die Wirkung von PEFT-Methoden erwies sich dabei als modellspezifisch. Des Weiteren wurde der positive Einfluss eines detaillierten, schema-erweiterten Promptings sowie spezieller Tokens und Loss-Maskierung auf die Genauigkeit beider Modellfamilien bestätigt.

Die Ergebnisse dieser Arbeit belegen die Trainierbarkeit leistungsfähiger Text-zu-SQL-Systeme auf Standard-GPUs und liefern differenzierte Erkenntnisse zur Wahl der Modellarchitektur und Feinabstimmungsmethode in Abhängigkeit von Genauigkeitsanforderungen und verfügbaren Ressourcen.

## Forschungsfragen

Diese Arbeit adressiert folgende Forschungsfragen:

* **1: Architektur- und Vortrainingsvergleich bei gleichem Adaptionsverfahren:** Leistet ein spezifisch für Code-Generierung vortrainiertes Decoder-Only-Modell (hier: DeepSeek-Coder-6.7B mit LoRA-Adaption) bei vergleichbarem Modellumfang eine höhere SQL-Genauigkeit als ein für allgemeine Sprachverständnis- und Generierungsaufgaben instruktions-vortrainiertes Encoder-Decoder-Modell (hier: FLAN-T5-Large mit LoRA-Adaption)?
* **2: Effizienz und Genauigkeit verschiedener PEFT-Methoden:** Welche konkreten Speicher- und Laufzeitvorteile erzielt eine fortschrittliche Parameter-effiziente Feinabstimmungsmethode wie QLoRA im Vergleich zu klassischem LoRA beim Training eines kompakten Modells (hier: DeepSeek-1.3B), und in welchem Ausmaß beeinflusst dies die erreichte Übersetzungsgenauigkeit?
* **3: Wirkung des Schema-Promptings auf unterschiedliche Architekturen:** Verbessert ein detailliertes, schema-erweitertes Prompting, das Tabellen- und Spaltentypinformationen sowie spezielle Start- und End-Tokens für die SQL-Generierung (`<SQL_START>`, `<SQL_END>`) beinhaltet, die Leistung beider untersuchten Modellfamilien (Encoder-Decoder T5 und Decoder-Only DeepSeek) in gleichem Maße, oder profitiert eine der Architekturen stärker von dieser Art der Kontexteinbettung?

## Methodik im Überblick

* **Modelle:**
    * T5-Familie: T5-base, T5-large, T5-v11-large, FLAN-T5-large 
    * DeepSeek-Familie: DeepSeek-Coder-1.3B-instruct, DeepSeek-Coder-6.7b-instruct
* **Techniken:**
    * Vollständiges Fine-Tuning
    * Low-Rank Adaptation (LoRA)
    * Quantized Low-Rank Adaptation (QLoRA)
    * Schema-angereichertes Prompting mit verschiedenen Detaillierungsgraden 
    * Spezielle Tokens (`<SQL_START>`, `<SQL_END>`) und Loss-Masking für DeepSeek-Modelle
* **Datensatz:** Spider Benchmark. Dies ist ein umfangreicher, domänenübergreifender Datensatz für komplexe Text-zu-SQL-Aufgaben.
    * **Wichtiger Hinweis:** Der Spider-Datensatz muss separat von der offiziellen Quelle bezogen werden: [https://yale-lily.github.io/spider](https://yale-lily.github.io/spider)
* **Evaluationsmetriken:** Primär Exact Match Accuracy (EM) und Execution Accuracy (EX), berechnet mit dem offiziellen Spider-Evaluationsskript. Zusätzlich wurden F1-Scores für einzelne SQL-Komponenten analysiert.
* **Hardware-Optimierung:** Einsatz von Gradient Checkpointing, Mixed Precision Training (BF16) und 4-Bit/8-Bit Quantisierung.

## Systemarchitektur

Die entwickelte Systemarchitektur für die Text-zu-SQL-Pipeline umfasst drei Hauptkomponenten:

1.  **Prompt-Builder:** Reichert die natürlichsprachliche Frage mit detaillierten Schema-Metadaten (Tabellen, Spalten, Typen, Schlüsselbeziehungen) an.
2.  **LLM-Backend:** Verarbeitet den angereicherten Prompt unter Verwendung der T5- oder DeepSeek-Modelle.
3.  **Adapter-Ebene (optional):** Injiziert trainierte LoRA/QLoRA-Gewichte in das Basismodell für Parameter-effizientes Fine-Tuning.

## Wichtigste Ergebnisse

* Instruktionsoptimierte Encoder-Decoder-Architekturen (FLAN-T5-large-FT) zeigten die höchste Genauigkeit (62% EX).
* Code-vortrainierte Decoder-Only Modelle (DeepSeek-6.7B-LoRA) erreichten trotz größerer Basisparameterzahl nicht die Genauigkeit der besten T5-Konfiguration.
* QLoRA verbesserte beim DeepSeek-1.3B Modell sowohl die Genauigkeit als auch die Ressourceneffizienz im Vergleich zum Full-Fine-Tuning signifikant (+7pp EX, -45% VRAM).
* Die Auswirkungen von Adapter-Methoden (LoRA/QLoRA) sind modellspezifisch: Leistungssteigerung bei DeepSeek-1.3B, Genauigkeitsverlust bei FLAN-T5-large im Vergleich zu Full-FT.
* Schema-angereichertes Prompting und spezielle Tokens verbesserten die Leistung über verschiedene Architekturen hinweg.

## Setup und Nutzung

Die Experimente wurden primär auf Google Colab unter Verwendung von NVIDIA A100 GPUs durchgeführt. Einige Experimente zur PICARD-Integration wurden lokal mittels Docker (Python 3.10) durchgeführt.

**Voraussetzungen (Beispiel für Trainingsskripte):**
* Python 3.9+
* PyTorch
* Hugging Face Transformers, Datasets, Accelerate, PEFT, BitsandBytes, TRL
* Zugang zum Spider-Datensatz (siehe oben)
