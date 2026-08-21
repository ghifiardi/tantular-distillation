// Score model edit output against the add-in's REAL contract parser.
//
//   node scripts/check_edit_contract.mjs <cases.json> <addin-src-dir> > result.json
//
// The `office_json_contract` gate as originally configured runs the add-in's own
// unit tests with fetch mocked — it never sees a model, so it cannot detect an
// adapter regression. This is the model-dependent half: it takes what the model
// actually produced and pushes it through the SAME functions the product uses.
//
// Deliberately imports from the add-in rather than reimplementing the contract.
// A reimplementation would drift, and then the gate would measure our copy of
// the rules instead of the rules the product enforces.
//
// Three measures, in increasing strength:
//
//   parse_ok        parseEditContract() accepted the raw output at all
//   fields_ok       every edit carries a usable find/replace/occurrence
//   contract_ok     resolveEdits() located every edit in the real document AND
//                   applyEditsToText() reported them applied
//
// contract_ok is the one that matters. A model can emit perfectly valid JSON
// whose `find` strings appear nowhere in the document; that parses, and is
// useless. Only locating and applying proves the edit contract still works.
//
// FAILS CLOSED: a missing cases file, a missing add-in source, or an unreadable
// module aborts with a non-zero exit rather than reporting an empty pass.

import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";

const [casesPath, addinSrc] = process.argv.slice(2);
if (!casesPath || !addinSrc) {
  console.error("usage: check_edit_contract.mjs <cases.json> <addin-src-dir>");
  process.exit(2);
}

let cases;
try {
  cases = JSON.parse(readFileSync(casesPath, "utf8"));
} catch (e) {
  console.error(`cannot read cases file ${casesPath}: ${e.message}`);
  process.exit(2);
}
if (!Array.isArray(cases) || cases.length === 0) {
  console.error("cases file is empty — refusing to report a vacuous pass");
  process.exit(2);
}

let parseEditContract, resolveEdits, applyEditsToText;
try {
  const contract = await import(
    pathToFileURL(resolve(addinSrc, "chat/editContract.js")).href);
  const apply = await import(
    pathToFileURL(resolve(addinSrc, "chat/applyEdits.js")).href);
  ({ parseEditContract, resolveEdits } = contract);
  ({ applyEditsToText } = apply);
} catch (e) {
  console.error(`cannot load the add-in parser from ${addinSrc}: ${e.message}`);
  process.exit(2);
}
if (typeof parseEditContract !== "function" || typeof applyEditsToText !== "function") {
  console.error("add-in module loaded but does not export the expected functions");
  process.exit(2);
}

const results = cases.map((c) => {
  const row = { id: c.id, parse_ok: false, fields_ok: false, contract_ok: false,
                edits: 0, error: null };
  if (typeof c.completion !== "string" || c.completion.trim() === "") {
    row.error = "no model output";     // missing output is a FAILURE, not a skip
    return row;
  }
  let parsed;
  try {
    parsed = parseEditContract(c.completion);
    row.parse_ok = true;
    row.edits = parsed.edits.length;
  } catch (e) {
    row.error = `parse: ${e.message}`;
    return row;
  }
  row.fields_ok = parsed.edits.every(
    (e) => typeof e.find === "string" && e.find.length > 0 &&
           typeof e.replace === "string" &&
           Number.isInteger(e.occurrence) && e.occurrence >= 1);
  if (!row.fields_ok) { row.error = "fields: an edit lacks a usable find/replace/occurrence"; return row; }

  try {
    const located = resolveEdits(c.document, parsed.edits);
    const unresolved = located.filter((r) => r.error);
    if (unresolved.length) {
      row.error = `resolve: ${unresolved.length}/${located.length} edit(s) not found in the document`;
      return row;
    }
    const applied = applyEditsToText(c.document, parsed.edits);
    const notApplied = (applied.perEditStatus || []).filter((s) => s !== "applied");
    if (notApplied.length) {
      row.error = `apply: ${notApplied.length} edit(s) did not apply (${[...new Set(notApplied)].join(", ")})`;
      return row;
    }
    row.contract_ok = true;
    // The APPLIED document, so a caller can check what the edit actually did:
    // which span changed, which spans must not have, and whether any figure or
    // name moved. contract_ok says the edit applied; it says nothing about
    // whether it applied to the right place or preserved what it must.
    row.applied = applied.text ?? applied.result ?? null;
    row.edits = parsed.edits.map((e) => ({ find: e.find, replace: e.replace,
                                           occurrence: e.occurrence }));
  } catch (e) {
    row.error = `apply: ${e.message}`;
  }
  return row;
});

const n = results.length;
const count = (k) => results.filter((r) => r[k]).length;
process.stdout.write(JSON.stringify({
  items: n,
  parse_ok: count("parse_ok"),
  fields_ok: count("fields_ok"),
  contract_ok: count("contract_ok"),
  rate: n ? count("contract_ok") / n : 0,
  _rate_is: "contract_ok / items — parsed AND located AND applied",
  results,
}, null, 2) + "\n");
