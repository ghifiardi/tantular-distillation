# Tantular Office

Asisten produktivitas Indonesian-first untuk Word, Excel, dan PowerPoint. Berjalan lokal, tanpa mengirim dokumen ke luar mesin.

Ini adalah **profil runtime** di atas Qwen3.5-9B (system prompt, template, parameter) — **bukan hasil fine-tune**. Bobotnya adalah Qwen3.5, Apache 2.0.

## Cara pakai

```
ollama pull ghifidanukusumo/tantular
```

## PENTING: gunakan /api/chat, bukan /v1/chat/completions

Model ini menghasilkan reasoning panjang bila thinking tidak dimatikan. Endpoint OpenAI-compatible milik Ollama **mengabaikan** `think: false` dan `chat_template_kwargs.enable_thinking`, sehingga jawaban bisa kosong tanpa pesan error.

Terukur pada satu tugas edit dokumen:

- lewat `/v1/chat/completions`: 21.808 karakter reasoning, **jawaban kosong**, 512 detik
- lewat `/api/chat` dengan `think: false`: **2 detik**, jawaban benar

```
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

Bila aplikasi Anda memakai bentuk OpenAI, terjemahkan di sisi klien: `max_tokens` menjadi `options.num_predict`, dan selalu kirim `think: false`.

## Tag yang tersedia

- **`latest`** dan **`q8-0.5`** — Q8_0, 10 GB. Default. Untuk RAM 16 GB ke atas.
- **`q4-0.4`** — Q4_K_M, 6.6 GB. Rollback, atau mesin dengan RAM lebih kecil.
- **`lite`** — Q4_K_M 4B, 3.4 GB. Untuk RAM di bawah 16 GB.

## Kualitas terukur

Diukur pada 60 item internal (40 voice, 20 edit-contract) melalui `/api/chat` dengan `think: false`, temperature 0, satu permintaan pada satu waktu, GPU Apple lokal.

Q8_0 (`latest`):

- Indonesian voice: 38 dari 40
- Edit-contract JSON: 19 dari 20
- Jawaban kosong: 0
- Latency p50 6,4 detik; p95 25,5 detik

Q4_K_M (`q4-0.4`):

- Indonesian voice: 37 dari 40
- Edit-contract JSON: 18 dari 20
- Jawaban kosong: 0
- Latency p50 3,8 detik; p95 15,3 detik

Q8 menjadi default karena Q4_K_M kehilangan kualitas; Q8 menyamai hasil bobot bf16 pada kedua evaluasi. Q4 lebih cepat dan lebih kecil — pilih Q4 bila kecepatan lebih penting daripada satu-dua item ketelitian.

Angka latency berasal dari satu mesin dan bukan janji kinerja. Permintaan paling lambat pada pengukuran kami mencapai 30 detik.

## Batasan yang diketahui

- Pada instruksi multi-edit, model kadang mengeluarkan satu `find` melebihi batas 200 karakter alih-alih beberapa edit terarah. Terjadi pada Q4, Q8, maupun bf16 — 1 dari 20 item.
- Evaluasi memakai dokumen sintetis, bukan dokumen Office nyata.
- Ini profil, bukan fine-tune: kemampuan dasarnya adalah kemampuan Qwen3.5-9B.

## Lisensi

Apache 2.0, mengikuti Qwen3.5.
