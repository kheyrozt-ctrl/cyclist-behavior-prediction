# TorchScript to ONNX Export Equivalence

## Status

The four packaged bus-stop ONNX models passed the export-equivalence audit on
2026-06-27. The audit used ONNX opset 17 and
`atol=0.0001, rtol=0.0001`.

| Model | Source TorchScript SHA-256 | ONNX SHA-256 | Maximum absolute error | Class agreement |
|---|---|---|---:|---|
| head | `a5e24d67faf92c5bb171b2b7af47da9a04f08d641aacf74acc8803af83274b4d` | `127e72bba26e0755a4c227f72dcbc96986e108533f6e1501323bf8e3b5ddcc61` | `1.6689301e-06` | yes |
| upper | `035412b7b44e781a3832d0a88d700dacb426da4c43f7bc0fe705b63b480dad60` | `348646143a69130c1c2ccf2a3a228a331e3899c563b7304011da76d78723728f` | `1.4305115e-06` | yes |
| leg | `e5981f9c03958f91064ea31ade8149f89ca309850d7f0e9614073b6d5d337f99` | `0c7208a5f128e96f098fc87ae27ba9a185bcc6a0604a99ce2b727d76155ad1c1` | `2.3841858e-06` | yes |
| stage2 | `a1182b092a25608aa6f3aed400af0df3d0dbde1460e9e12b90f6f14a995b9622` | `a0211690f376ee7024051a28ef28280e7252f84d85b9eb80b45c0022ec12df77` | `3.3378601e-06` | yes |

The random end-to-end Stage1-feature-to-Stage2 check had maximum absolute error
`2.1457672e-06` with identical class indices.

## Public pose-data check

The audit also sampled 32 sequences from the public Hugging Face release
(99,734 frame rows, 559 sequences). Missing coordinates were imputed per
sequence with linear interpolation and edge fill.

- Stage1 maximum absolute errors:
  `head=3.7357211e-05`, `upper=2.1457672e-06`,
  `leg=3.8146973e-06`
- Stage2 maximum absolute error: `1.9073486e-06`
- class indices: identical

## Reproduction

Install `requirements-export-audit.txt`, then run:

```bash
python tools/audit_model_export.py --output-dir model-export-audit
```

The audit checks source hashes, exports all four models, validates ONNX graphs,
compares TorchScript and ONNX Runtime outputs across random and boundary inputs,
and writes a machine-readable JSON report.

## Boundary

This establishes CPU numerical equivalence for the tested inputs. It does not
establish TensorRT compatibility, FP16/INT8 equivalence, target-device latency,
or dataset-level prediction accuracy.
