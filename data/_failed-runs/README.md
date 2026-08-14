# Failed runs — NOT corpora

Partial output from runs that did not complete. Retained for diagnosis, never
merged into a corpus.

## expanded-attempt1-tunnel-drop

260-family run, 2026-08-14. The SSH tunnel to ai19 died at prompt index 19.

  19/260 traces written
  241 infrastructure failures (237 ConnectError, 4 RemoteProtocolError)
  0/3 and 0/3 on the two volatile replicate runs

Not merged, not resumed into a result set. A corpus assembled from a partial
run plus a later completion would mix two teacher sessions, and the split
balance and mode statistics would describe neither.

The tunnel dropping mid-run is now a REPEATED failure mode this session, not a
one-off: it also killed an earlier smoke attempt. A ~100 minute run over a
single long-lived SSH forward with no resume capability will keep failing this
way.
