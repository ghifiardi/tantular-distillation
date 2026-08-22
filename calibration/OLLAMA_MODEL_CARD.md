# Tantular Office

Asisten produktivitas **Indonesian-first** untuk Word, Excel, dan PowerPoint.
Berjalan lokal, tanpa mengirim dokumen ke luar mesin.

Ini adalah **profil runtime** di atas Qwen3.5-9B — system prompt, template, dan
parameter — **bukan hasil fine-tune**. Bobotnya adalah Qwen3.5 (Apache 2.0).

```bash
ollama pull ghifidanukusumo/tantular
```

## PENTING: gunakan `/api/chat`, bukan `/v1/chat/completions`

Model ini menghasilkan reasoning panjang jika thinking tidak dimatikan.
Endpoint OpenAI-compatible milik Ollama **mengabaikan** `think: false` maupun
`chat_template_kwargs.enable_thinking`, sehingga jawaban bisa kosong.

Terukur: satu tugas edit lewat `/v1/chat/completions` menghasilkan 21.808
karakter reasoning, **jawaban kosong**, setelah 512 detik. Tugas yang sama lewat
`/api/chat` dengan `think: false`: **2 detik**, jawaban benar.

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "ghifidanukusumo/tantular",
  "think": false,
  "stream": false,
  "options": { "temperature": 0.2, "num_predict": 1200 },
  "messages": [
    { "role": "user", "content": "Ringkas memo ini dalam tiga butir: ..." }
  ]
}'
```

Jika aplikasi Anda memerlukan bentuk OpenAI, terjemahkan di sisi klien:
`max_tokens` → `options.num_predict`, dan selalu kirim `think: false`.

## Tag

| tag | presisi | ukuran | untuk |
|---|---|---|---|
| `latest` / `q8-0.5` | Q8_0 | 10 GB | **default**; RAM ≥ 16 GB |
| `q4-0.4` | Q4_K_M | 6.6 GB | rollback, atau mesin RAM lebih kecil |
| `lite` | Q4_K_M (4B) | 3.4 GB | mesin RAM < 16 GB |

## Kualitas terukur

Dievaluasi pada 60 item buatan sendiri (40 voice, 20 edit-contract) melalui
`/api/chat` dengan `think: false`, `temperature 0`, satu permintaan pada satu
waktu, Apple GPU lokal.

| ukuran | Q8_0 (`latest`) | Q4_K_M (`q4-0.4`) |
|---|---|---|
| Indonesian voice | **38/40** | 37/40 |
| Edit-contract JSON | **19/20** | 18/20 |
| Jawaban kosong | 0 | 0 |
| Latency p50 / p95 | 6.4 s / 25.5 s | 3.8 s / 15.3 s |

**Q8 dipilih sebagai default karena Q4_K_M kehilangan kualitas** — Q8 menyamai
hasil bobot bf16 pada kedua evaluasi. Q4 lebih cepat dan lebih kecil; pilih Q4
bila kecepatan lebih penting daripada satu-dua item ketelitian.

Angka latency berasal dari satu mesin dan bukan janji kinerja. Permintaan
terlambat pada pengukuran kami mencapai 30 detik.

## Batasan yang diketahui

- Pada instruksi multi-edit, model kadang mengeluarkan satu `find` yang lebih
  panjang dari batas 200 karakter alih-alih beberapa edit terarah. Terjadi pada
  Q4, Q8, maupun bf16 — 1 dari 20 item.
- Evaluasi di atas memakai dokumen sintetis, bukan dokumen Office nyata.
- Ini profil, bukan fine-tune: kemampuan dasarnya adalah kemampuan Qwen3.5-9B.

## Lisensi

Apache 2.0, mengikuti Qwen3.5.
