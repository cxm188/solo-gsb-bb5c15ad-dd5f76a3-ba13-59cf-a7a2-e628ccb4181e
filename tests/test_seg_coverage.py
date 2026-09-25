"""GET /jobs/{id}/instances/{sop_uid}/seg-coverage：BINARY SEG 1-bit 掩码覆盖率。

覆盖：逐帧段号/源关联、多帧逐像素并集、MSB-first 1-bit 解码、源尺寸校验、
404/409 各类冲突一次性给出且不回显原始 UID。
"""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pydicom
import pytest
from fastapi.testclient import TestClient
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    RLELossless,
    RTImageStorage,
    SegmentationStorage,
)

from app.crypto import derive_uid
from tests.sample_data import _write_bytes

SECRET = "seg-coverage-secret"

STUDY = "6.1"
SERIES = "6.2"
FRAME = "6.3"
SRC1 = "6.10"
SRC2 = "6.11"
SEG_UID = "6.20"


# --------------------------------------------------------------------------- #
# 合成 DICOM
# --------------------------------------------------------------------------- #
def _pack_1bit(rows: int, columns: int, ones) -> bytes:
    """按 DICOM PS3.5 §8.1.1 打包：MSB-first，行末补零到字节边界。"""
    bpr = (columns + 7) // 8
    arr = np.zeros((rows, bpr), dtype=np.uint8)
    for r, c in ones:
        arr[r, c // 8] |= 0x80 >> (c % 8)
    return arr.tobytes()


def _source_bytes(
    sop_uid: str,
    *,
    rows: int = 4,
    columns: int = 8,
    sop_class=CTImageStorage,
    modality: str = "CT",
    pixel_bytes: bytes | None = None,
    bits_allocated: int = 16,
    transfer_syntax=ExplicitVRLittleEndian,
):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = transfer_syntax
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = STUDY
    ds.SeriesInstanceUID = SERIES
    ds.FrameOfReferenceUID = FRAME
    ds.PatientID = "P-1"
    ds.PatientName = "Zhang^San"
    ds.Modality = modality
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = columns
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_allocated
    ds.HighBit = bits_allocated - 1
    ds.PixelRepresentation = 0
    if pixel_bytes is None:
        pixel_bytes = bytes(rows * columns * (bits_allocated // 8))
    ds.PixelData = pixel_bytes
    return _write_bytes(ds)


def _source_image_item(source_sop: str, *, sop_class=CTImageStorage) -> Dataset:
    item = Dataset()
    item.ReferencedSOPClassUID = sop_class
    item.ReferencedSOPInstanceUID = source_sop
    return item


def _per_frame_item(
    segment_number,
    source_sops=None,
    *,
    direct_source: bool = False,
    multiple_derivations: bool = False,
):
    """构造逐帧功能组：段号 + DerivationImageSequence/SourceImageSequence 源引用。"""
    fg = Dataset()
    if segment_number is not None:
        sid = Dataset()
        if segment_number != "_omit_value":
            sid.ReferencedSegmentNumber = segment_number
        fg.SegmentIdentificationSequence = [sid]
    if source_sops is None:
        return fg
    if direct_source:
        fg.SourceImageSequence = [_source_image_item(uid) for uid in source_sops]
        return fg
    if multiple_derivations:
        # 每个 DerivationImageSequence 项必须是独立 Dataset（pydicom 按引用挂载）
        fg.DerivationImageSequence = []
        for uid in source_sops:
            der = Dataset()
            der.SourceImageSequence = [_source_image_item(uid)]
            fg.DerivationImageSequence.append(der)
    else:
        der = Dataset()
        der.SourceImageSequence = [_source_image_item(uid) for uid in source_sops]
        fg.DerivationImageSequence = [der]
    return fg


def _seg_bytes(
    sop_uid: str = SEG_UID,
    *,
    frames=((1, SRC1, ((0, 0),)),),
    rows: int = 4,
    columns: int = 8,
    sop_class=SegmentationStorage,
    transfer_syntax=ExplicitVRLittleEndian,
    segmentation_type: str | None = "BINARY",
    bits_allocated: int = 1,
    samples_per_pixel=1,
    photometric: str = "MONOCHROME2",
    pixel_repr: int = 0,
    number_of_frames=None,
    segment_numbers=(1,),
    per_frame_items=None,
    raw_pixel: bytes | None = None,
    include_pixel: bool = True,
    drop: tuple[str, ...] = (),
):
    """构造 SEG 实例。

    frames: (段号, 源 SOP, 前景像素坐标列表) 的帧列表；
    per_frame_items 给出时直接使用（用于构造畸形功能组）；
    raw_pixel 给出时逐帧元组里的掩码仍按 frames 顺序生成但整体被替换。
    """
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = transfer_syntax
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = STUDY
    ds.SeriesInstanceUID = SERIES
    ds.FrameOfReferenceUID = FRAME
    ds.PatientID = "P-1"
    ds.PatientName = "Zhang^San"
    ds.Modality = "SEG"
    ds.SamplesPerPixel = samples_per_pixel
    ds.PhotometricInterpretation = photometric
    ds.Rows = rows
    ds.Columns = columns
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_allocated
    ds.HighBit = bits_allocated - 1
    ds.PixelRepresentation = pixel_repr
    if segmentation_type is not None:
        ds.SegmentationType = segmentation_type
    declared_frames = len(frames) if number_of_frames is None else number_of_frames
    if declared_frames is not None:
        ds.NumberOfFrames = declared_frames
    if segment_numbers is not None:
        ds.SegmentSequence = []
        for n in segment_numbers:
            segment = Dataset()
            segment.SegmentNumber = n
            segment.SegmentLabel = f"SEG{n}"
            ds.SegmentSequence.append(segment)
    if per_frame_items is None:
        per_frame_items = []
        for segn, source_sop, ones in frames:
            sources = None if source_sop is None else [source_sop]
            per_frame_items.append(_per_frame_item(segn, sources))
    if per_frame_items:
        ds.PerFrameFunctionalGroupsSequence = per_frame_items
    if include_pixel:
        if raw_pixel is not None:
            pixel = raw_pixel
        else:
            pixel = b"".join(
                _pack_1bit(rows, columns, ones) for _, _, ones in frames
            )
        ds.PixelData = pixel
    for keyword in drop:
        if keyword in ds:
            del ds[keyword]
    return _write_bytes(ds)


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in files.items():
            zf.writestr(name, blob)
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DICOM_ANON_WORK_DIR", str(tmp_path / "jobs"))
    import importlib
    import app.config
    import app.jobs
    importlib.reload(app.config)
    importlib.reload(app.jobs)
    import app.main
    importlib.reload(app.main)
    with TestClient(app.main.app) as c:
        yield c


def _upload(client, files: dict[str, bytes]):
    r = client.post(
        "/jobs",
        files={"file": ("studies.zip", _zip_bytes(files), "application/zip")},
        data={"secret": SECRET, "shift_dates": "false"},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _anon(original: str) -> str:
    return derive_uid(SECRET, original)


def _seg_coverage(client, job_id: str, sop_uid: str = SEG_UID):
    return client.get(
        f"/jobs/{job_id}/instances/{_anon(sop_uid)}/seg-coverage"
    )


def _basic_files(seg_blob: bytes, *, sources=(SRC1,), source_kwargs=None):
    files = {}
    for i, source_uid in enumerate(sources):
        kwargs = (source_kwargs or {}).get(source_uid, {})
        files[f"src{i}.dcm"] = _source_bytes(source_uid, **kwargs)
    files["seg.dcm"] = seg_blob
    return files


# --------------------------------------------------------------------------- #
# 成功路径
# --------------------------------------------------------------------------- #
def test_basic_single_frame_coverage(client):
    seg = _seg_bytes(frames=((1, SRC1, ((0, 0), (1, 1), (3, 7))),))
    job = _upload(client, _basic_files(seg))
    r = _seg_coverage(client, job["id"])
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["sop_instance_uid"] == _anon(SEG_UID)
    assert data["output_file"] == f"{_anon(SEG_UID)}.dcm"
    assert data["rows"] == 4 and data["columns"] == 8
    assert data["number_of_frames"] == 1
    assert data["item_count"] == 1
    item = data["items"][0]
    assert item["source_sop_instance_uid"] == _anon(SRC1)
    assert item["segment_number"] == 1
    assert item["frame_count"] == 1
    assert item["foreground_pixel_count"] == 3
    assert item["image_pixel_count"] == 32
    assert item["coverage_ratio"] == pytest.approx(3 / 32)


def test_multiple_frames_same_source_and_segment_are_unioned(client):
    # 两帧前景重叠一个像素：并集应为 3 而不是 4
    seg = _seg_bytes(frames=(
        (1, SRC1, ((0, 0), (1, 1))),
        (1, SRC1, ((0, 0), (2, 2))),
    ))
    job = _upload(client, _basic_files(seg))
    data = _seg_coverage(client, job["id"]).json()
    assert data["item_count"] == 1
    item = data["items"][0]
    assert item["frame_count"] == 2
    assert item["foreground_pixel_count"] == 3
    assert item["coverage_ratio"] == pytest.approx(3 / 32)


def test_items_sorted_by_source_uid_then_segment(client):
    seg = _seg_bytes(
        segment_numbers=(1, 2),
        frames=(
            (2, SRC1, ((0, 0),)),
            (1, SRC2, ((0, 0),)),
            (1, SRC1, ((0, 0),)),
            (2, SRC2, ((0, 0),)),
            (1, SRC1, ((1, 1),)),
        ),
    )
    job = _upload(client, _basic_files(seg, sources=(SRC1, SRC2)))
    data = _seg_coverage(client, job["id"]).json()
    keys = [(it["source_sop_instance_uid"], it["segment_number"])
            for it in data["items"]]
    assert keys == sorted(keys)
    # 四种 (源, 段) 组合齐全；源 UID 顺序按匿名 UID 字典序而非原 UID
    expected_sources = sorted([_anon(SRC1), _anon(SRC2)])
    assert keys == [
        (expected_sources[0], 1),
        (expected_sources[0], 2),
        (expected_sources[1], 1),
        (expected_sources[1], 2),
    ]
    by_key = {(it["source_sop_instance_uid"], it["segment_number"]): it
              for it in data["items"]}
    assert by_key[(_anon(SRC1), 1)]["frame_count"] == 2
    assert by_key[(_anon(SRC1), 1)]["foreground_pixel_count"] == 2
    for item in data["items"]:
        assert item["image_pixel_count"] == 32
        assert item["coverage_ratio"] == pytest.approx(
            item["foreground_pixel_count"] / 32
        )


def test_msb_first_packing(client):
    """单字节行：第 1 个像素对应 0x80，最后一个像素对应 0x01。"""
    seg_first = _seg_bytes(frames=((1, SRC1, ((0, 0),)),))
    seg_last = _seg_bytes(frames=((1, SRC1, ((0, 7),)),))
    job1 = _upload(client, _basic_files(seg_first))
    data1 = _seg_coverage(client, job1["id"]).json()
    assert data1["items"][0]["foreground_pixel_count"] == 1

    job2 = _upload(client, _basic_files(seg_last))
    data2 = _seg_coverage(client, job2["id"]).json()
    assert data2["items"][0]["foreground_pixel_count"] == 1


def test_full_mask_and_empty_mask(client):
    full = {(r, c) for r in range(4) for c in range(8)}
    seg_full = _seg_bytes(frames=((1, SRC1, tuple(sorted(full))),))
    job = _upload(client, _basic_files(seg_full))
    item = _seg_coverage(client, job["id"]).json()["items"][0]
    assert item["foreground_pixel_count"] == 32
    assert item["coverage_ratio"] == 1.0

    seg_empty = _seg_bytes(frames=((1, SRC1, ()),))
    job2 = _upload(client, _basic_files(seg_empty))
    item2 = _seg_coverage(client, job2["id"]).json()["items"][0]
    assert item2["foreground_pixel_count"] == 0
    assert item2["coverage_ratio"] == 0.0


def test_non_byte_aligned_columns(client):
    # 6 列：每行仍占 1 字节，第 7/8 位为行内补齐位，必须不计入像素
    seg = _seg_bytes(
        rows=2,
        columns=6,
        frames=((1, SRC1, ((0, 0), (1, 5))),),
    )
    sources = {SRC1: {"rows": 2, "columns": 6}}
    job = _upload(client, _basic_files(seg, source_kwargs=sources))
    data = _seg_coverage(client, job["id"]).json()
    assert data["rows"] == 2 and data["columns"] == 6
    item = data["items"][0]
    assert item["image_pixel_count"] == 12
    assert item["foreground_pixel_count"] == 2
    assert item["coverage_ratio"] == pytest.approx(2 / 12)


def test_direct_source_image_sequence_on_functional_group(client):
    # 无 DerivationImageSequence，源直接挂在逐帧功能组项的 SourceImageSequence
    fg = _per_frame_item(1, [SRC1], direct_source=True)
    seg = _seg_bytes(
        frames=((1, SRC1, ((0, 0),)),),
        per_frame_items=[fg],
    )
    job = _upload(client, _basic_files(seg))
    r = _seg_coverage(client, job["id"])
    assert r.status_code == 200, r.text
    assert r.json()["items"][0]["source_sop_instance_uid"] == _anon(SRC1)


def test_duplicate_source_uid_in_same_frame_counts_once(client):
    # 同一帧的两个 DerivationImageSequence 项指向同一个源 UID：不算歧义
    fg = _per_frame_item(1, [SRC1, SRC1], multiple_derivations=True)
    seg = _seg_bytes(
        frames=((1, SRC1, ((0, 0),)),),
        per_frame_items=[fg],
    )
    job = _upload(client, _basic_files(seg))
    r = _seg_coverage(client, job["id"])
    assert r.status_code == 200, r.text
    assert r.json()["items"][0]["frame_count"] == 1


def test_source_can_be_non_ct_image(client):
    # 源引用只要求是任务内唯一成功输出的图像实例（这里用 RT Image Storage）
    seg = _seg_bytes(frames=((1, SRC1, ((0, 0),)),))
    files = {
        "ri.dcm": _source_bytes(
            SRC1, sop_class=RTImageStorage, modality="RTIMAGE"
        ),
        "seg.dcm": seg,
    }
    job = _upload(client, files)
    r = _seg_coverage(client, job["id"])
    assert r.status_code == 200, r.text


def test_implicit_vr_little_endian_seg(client):
    seg = _seg_bytes(
        frames=((1, SRC1, ((0, 0),)),),
        transfer_syntax=ImplicitVRLittleEndian,
    )
    job = _upload(client, _basic_files(seg))
    r = _seg_coverage(client, job["id"])
    assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- #
# 404
# --------------------------------------------------------------------------- #
def test_job_not_found_404(client):
    r = client.get(f"/jobs/deadbeef/instances/{_anon(SEG_UID)}/seg-coverage")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["code"] == "JOB_NOT_FOUND"
    assert detail["sop_instance_uid"] == _anon(SEG_UID)


def test_instance_not_found_404(client):
    job = _upload(client, _basic_files(_seg_bytes()))
    r = client.get(
        f"/jobs/{job['id']}/instances/{_anon('9.9.999')}/seg-coverage"
    )
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["code"] == "INSTANCE_NOT_FOUND"
    assert detail["sop_instance_uid"] == _anon("9.9.999")


def test_seg_output_file_removed_404(client):
    job = _upload(client, _basic_files(_seg_bytes()))
    from app.main import registry
    job_obj = registry.get(job["id"])
    os.remove(os.path.join(job_obj.result.output_dir, f"{_anon(SEG_UID)}.dcm"))
    r = _seg_coverage(client, job["id"])
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["code"] == "INSTANCE_OUTPUT_MISSING"
    assert detail["reason"] == "output_missing"
    assert detail["output_file"] == f"{_anon(SEG_UID)}.dcm"


def test_failed_job_409(client):
    r = client.post(
        "/jobs",
        files={"file": ("broken.zip", b"PK\x03\x04 garbage", "application/zip")},
        data={"secret": SECRET},
    )
    job = r.json()
    assert job["status"] == "failed"
    resp = client.get(f"/jobs/{job['id']}/instances/x/seg-coverage")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "JOB_NOT_READY"
    assert detail["sop_instance_uid"] == "x"


# --------------------------------------------------------------------------- #
# 409：SEG 结构 / 传输语法 / 像素元数据
# --------------------------------------------------------------------------- #
def _conflict(client, files, seg_uid: str = SEG_UID):
    job = _upload(client, files)
    r = client.get(
        f"/jobs/{job['id']}/instances/{_anon(seg_uid)}/seg-coverage"
    )
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "SEG_COVERAGE_CONFLICT"
    assert detail["conflict_count"] == len(detail["problems"])
    return detail


def _find(detail, reason):
    return [p for p in detail["problems"] if p["reason"] == reason]


def test_conflict_ambiguous_seg_output(client):
    # 两个文件共享同一 SOPInstanceUID -> <uid>.dcm / <uid>-2.dcm
    files = {
        "src.dcm": _source_bytes(SRC1),
        "seg_a.dcm": _seg_bytes(),
        "seg_b.dcm": _seg_bytes(
            frames=((1, SRC1, ((0, 0), (0, 1))),)
        ),
    }
    job = _upload(client, files)
    r = _seg_coverage(client, job["id"])
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "AMBIGUOUS_INSTANCE_OUTPUT"
    assert detail["reason"] == "ambiguous_output"
    assert detail["output_files"] == sorted(
        [f"{_anon(SEG_UID)}.dcm", f"{_anon(SEG_UID)}-2.dcm"]
    )


def test_conflict_seg_unreadable(client):
    job = _upload(client, _basic_files(_seg_bytes()))
    from app.main import registry
    job_obj = registry.get(job["id"])
    target = os.path.join(job_obj.result.output_dir, f"{_anon(SEG_UID)}.dcm")
    with open(target, "wb") as fh:
        fh.write(b"\x00\x01not dicom anymore")
    r = _seg_coverage(client, job["id"])
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "SEG_COVERAGE_CONFLICT"
    assert _find(detail, "unreadable")


def test_conflict_not_segmentation_storage(client):
    # SOPClassUID 是 CT Image Storage 但按 SEG 路径请求
    blob = _seg_bytes(sop_class=CTImageStorage)
    detail = _conflict(client, _basic_files(blob))
    problems = _find(detail, "not_segmentation")
    assert len(problems) == 1
    assert problems[0]["sop_class_uid"] == CTImageStorage


def test_conflict_compressed_transfer_syntax(client):
    # RLE 封装传输语法：本接口不解码封装像素（封装帧内容本身无需是合法 RLE 码流，
    # 传输语法校验先于像素解码即可拒绝）
    from pydicom.encaps import encapsulate

    blob = _seg_bytes()
    ds = pydicom.dcmread(io.BytesIO(blob))
    frame_bytes = bytes(ds.PixelData)
    ds.file_meta.TransferSyntaxUID = RLELossless
    ds.PixelData = encapsulate([frame_bytes])
    blob = _write_bytes(ds)
    detail = _conflict(client, _basic_files(blob))
    problems = _find(detail, "unsupported_transfer_syntax")
    assert len(problems) == 1
    assert problems[0]["transfer_syntax_uid"] == RLELossless


def test_conflict_fractional_segmentation_type(client):
    blob = _seg_bytes(segmentation_type="FRACTIONAL", bits_allocated=8)
    detail = _conflict(client, _basic_files(blob))
    problems = _find(detail, "unsupported_segmentation_type")
    assert len(problems) == 1
    assert problems[0]["segmentation_type"] == "FRACTIONAL"


def test_conflict_missing_segmentation_type(client):
    blob = _seg_bytes(segmentation_type=None)
    detail = _conflict(client, _basic_files(blob))
    assert _find(detail, "unsupported_segmentation_type")


def test_conflict_non_binary_type_and_wrong_bits_collected_together(client):
    blob = _seg_bytes(
        segmentation_type="FRACTIONAL", bits_allocated=8
    )
    detail = _conflict(client, _basic_files(blob))
    assert _find(detail, "unsupported_segmentation_type")
    assert _find(detail, "invalid_pixel_metadata")  # BitsAllocated != 1


def test_conflict_missing_pixel_data(client):
    blob = _seg_bytes(include_pixel=False)
    detail = _conflict(client, _basic_files(blob))
    assert _find(detail, "missing_pixel_data")


def test_conflict_invalid_rows_columns(client):
    # Rows=0：源仍为 4×8，SEG 元数据本身非法
    blob = _seg_bytes(rows=4, columns=8, drop=("Rows",))
    ds = pydicom.dcmread(io.BytesIO(blob))
    ds.Rows = 0
    blob = _write_bytes(ds)
    detail = _conflict(client, _basic_files(blob))
    problems = _find(detail, "invalid_pixel_metadata")
    assert problems  # Rows/Columns 无效


def test_conflict_number_of_frames_invalid(client):
    blob = _seg_bytes(number_of_frames=0, frames=())
    detail = _conflict(client, _basic_files(blob))
    problems = _find(detail, "invalid_pixel_metadata")
    assert any("NumberOfFrames" in p["message"] for p in problems)


def test_conflict_missing_segment_sequence(client):
    blob = _seg_bytes(segment_numbers=None)
    detail = _conflict(client, _basic_files(blob))
    assert _find(detail, "invalid_pixel_metadata")


def test_conflict_duplicate_segment_number_in_segment_sequence(client):
    blob = _seg_bytes(segment_numbers=(1, 1))
    detail = _conflict(client, _basic_files(blob))
    problems = _find(detail, "invalid_pixel_metadata")
    assert any(p.get("segment_number") == 1 for p in problems)


def test_conflict_missing_per_frame_functional_groups(client):
    blob = _seg_bytes(per_frame_items=[])
    detail = _conflict(client, _basic_files(blob))
    assert _find(detail, "frame_functional_groups_missing")


def test_conflict_frame_count_mismatch(client):
    # 声明 2 帧，逐帧序列只有 1 项，像素也只有 1 帧
    frames = ((1, SRC1, ((0, 0),)),)
    blob = _seg_bytes(frames=frames, number_of_frames=2)
    detail = _conflict(client, _basic_files(blob))
    problems = _find(detail, "frame_count_mismatch")
    assert len(problems) == 1
    assert problems[0]["declared_number_of_frames"] == "2"
    assert problems[0]["per_frame_item_count"] == 1


def test_conflict_pixel_data_too_short(client):
    # 1 帧 4 行 × 1 字节/行需要 4 字节；DICOM 写出会补齐到偶数，这里补到 2 仍不足
    blob = _seg_bytes(raw_pixel=b"\x00\x00")
    detail = _conflict(client, _basic_files(blob))
    problems = _find(detail, "invalid_pixel_data_length")
    assert len(problems) == 1
    assert problems[0]["pixel_data_length"] == 2
    assert problems[0]["expected_pixel_data_length"] == 4


def test_conflict_pixel_data_too_long(client):
    blob = _seg_bytes(raw_pixel=b"\x00" * 9)  # 期望 4 字节，奇偶补齐也只允许 5
    detail = _conflict(client, _basic_files(blob))
    assert _find(detail, "invalid_pixel_data_length")


def test_odd_expected_pixel_length_accepts_one_padding_byte(client):
    # 3 行 × 1 字节/行 = 3（奇数），PixelData 偶数字节补齐为 4 字节
    mask = _pack_1bit(3, 8, ((0, 0),))
    seg = _seg_bytes(
        rows=3,
        columns=8,
        frames=((1, SRC1, ((0, 0),)),),
        raw_pixel=mask + b"\x00",
    )
    sources = {SRC1: {"rows": 3, "columns": 8}}
    job = _upload(client, _basic_files(seg, source_kwargs=sources))
    r = _seg_coverage(client, job["id"])
    assert r.status_code == 200, r.text
    assert r.json()["items"][0]["foreground_pixel_count"] == 1


def test_conflict_photometric_rgb(client):
    blob = _seg_bytes(samples_per_pixel=3, photometric="RGB")
    detail = _conflict(client, _basic_files(blob))
    reasons = {p["reason"] for p in detail["problems"]}
    assert "invalid_pixel_metadata" in reasons


# --------------------------------------------------------------------------- #
# 409：逐帧段号 / 源引用
# --------------------------------------------------------------------------- #
def test_conflict_frame_missing_segment_identification(client):
    fg = _per_frame_item(None, [SRC1])
    blob = _seg_bytes(
        frames=((1, SRC1, ((0, 0),)),),
        per_frame_items=[fg],
    )
    detail = _conflict(client, _basic_files(blob))
    problems = _find(detail, "missing_segment_number")
    assert len(problems) == 1
    assert problems[0]["frame_index"] == 0


def test_conflict_frame_empty_referenced_segment_number(client):
    fg = _per_frame_item("_omit_value", [SRC1])
    blob = _seg_bytes(
        frames=((1, SRC1, ((0, 0),)),),
        per_frame_items=[fg],
    )
    detail = _conflict(client, _basic_files(blob))
    assert _find(detail, "missing_segment_number")


def test_conflict_frame_segment_number_not_in_segment_sequence(client):
    blob = _seg_bytes(
        segment_numbers=(1,),
        frames=((2, SRC1, ((0, 0),)),),
    )
    detail = _conflict(client, _basic_files(blob))
    problems = _find(detail, "unknown_segment_number")
    assert len(problems) == 1
    assert problems[0]["frame_index"] == 0
    assert problems[0]["segment_number"] == 2


def test_conflict_frame_without_any_source_sequence(client):
    fg = _per_frame_item(1, None)
    blob = _seg_bytes(
        frames=((1, SRC1, ((0, 0),)),),
        per_frame_items=[fg],
    )
    detail = _conflict(client, _basic_files(blob))
    problems = _find(detail, "missing_source_reference")
    assert len(problems) == 1
    assert problems[0]["frame_index"] == 0
    assert problems[0]["segment_number"] == 1


def test_conflict_source_sequence_without_uid(client):
    der = Dataset()
    der.SourceImageSequence = [Dataset()]  # 无 ReferencedSOPInstanceUID
    fg = Dataset()
    sid = Dataset()
    sid.ReferencedSegmentNumber = 1
    fg.SegmentIdentificationSequence = [sid]
    fg.DerivationImageSequence = [der]
    blob = _seg_bytes(
        frames=((1, SRC1, ((0, 0),)),),
        per_frame_items=[fg],
    )
    detail = _conflict(client, _basic_files(blob))
    assert _find(detail, "missing_source_reference")


def test_conflict_frame_references_two_distinct_sources(client):
    fg = _per_frame_item(1, [SRC1, SRC2], multiple_derivations=True)
    blob = _seg_bytes(
        frames=((1, SRC1, ((0, 0),)),),
        per_frame_items=[fg],
    )
    files = {
        "src1.dcm": _source_bytes(SRC1),
        "src2.dcm": _source_bytes(SRC2),
        "seg.dcm": blob,
    }
    detail = _conflict(client, files)
    problems = _find(detail, "ambiguous_source_reference")
    assert len(problems) == 1
    assert problems[0]["frame_index"] == 0
    assert problems[0]["referenced_sop_instance_uids"] == sorted(
        [_anon(SRC1), _anon(SRC2)]
    )


def test_conflict_source_not_in_job_output(client):
    # SEG 引用的源 SOP 未随 ZIP 上传（dangling）：匿名化一致重写但输出中不存在
    blob = _seg_bytes(frames=((1, "6.999", ((0, 0),)),))
    detail = _conflict(client, _basic_files(blob))
    problems = _find(detail, "source_not_in_output")
    assert len(problems) == 1
    assert problems[0]["frame_index"] == 0
    assert problems[0]["source_sop_instance_uid"] == _anon("6.999")


def test_conflict_ambiguous_source_output(client):
    # 源 SOP 在输入 ZIP 内重复 -> 输出 <uid>.dcm / <uid>-2.dcm
    files = {
        "dup_a.dcm": _source_bytes(SRC1),
        "dup_b.dcm": _source_bytes(SRC1),
        "seg.dcm": _seg_bytes(),
    }
    detail = _conflict(client, files)
    problems = _find(detail, "ambiguous_source_output")
    assert len(problems) == 1
    assert problems[0]["source_sop_instance_uid"] == _anon(SRC1)
    assert problems[0]["output_files"] == sorted(
        [f"{_anon(SRC1)}.dcm", f"{_anon(SRC1)}-2.dcm"]
    )


def test_conflict_collects_all_frame_problems_at_once(client):
    # 第 0 帧段号未知；第 1 帧无源引用：同一次 409 全部给出
    f0 = _per_frame_item(9, [SRC1])
    f1 = _per_frame_item(1, None)
    blob = _seg_bytes(
        frames=(
            (1, SRC1, ((0, 0),)),
            (1, SRC1, ((0, 1),)),
        ),
        per_frame_items=[f0, f1],
    )
    detail = _conflict(client, _basic_files(blob))
    reasons = {p["reason"] for p in detail["problems"]}
    assert "unknown_segment_number" in reasons
    assert "missing_source_reference" in reasons
    # 帧问题按帧序号排序
    assert [p["frame_index"] for p in detail["problems"]] == sorted(
        p["frame_index"] for p in detail["problems"]
    )


# --------------------------------------------------------------------------- #
# 409：源图像
# --------------------------------------------------------------------------- #
def test_conflict_source_dimension_mismatch(client):
    sources = {SRC1: {"rows": 8, "columns": 8}}  # SEG 是 4×8
    detail = _conflict(client, _basic_files(_seg_bytes(), source_kwargs=sources))
    problems = _find(detail, "source_dimension_mismatch")
    assert len(problems) == 1
    assert problems[0]["frame_index"] == 0
    assert problems[0]["source_rows"] == 8
    assert problems[0]["source_columns"] == 8
    assert problems[0]["seg_rows"] == 4
    assert problems[0]["seg_columns"] == 8
    assert problems[0]["source_sop_instance_uid"] == _anon(SRC1)


def test_conflict_source_output_missing(client):
    job = _upload(client, _basic_files(_seg_bytes()))
    from app.main import registry
    job_obj = registry.get(job["id"])
    os.remove(os.path.join(job_obj.result.output_dir, f"{_anon(SRC1)}.dcm"))
    r = _seg_coverage(client, job["id"])
    assert r.status_code == 409
    detail = r.json()["detail"]
    problems = _find(detail, "source_output_missing")
    assert len(problems) == 1
    assert problems[0]["source_sop_instance_uid"] == _anon(SRC1)


def test_conflict_source_unreadable(client):
    job = _upload(client, _basic_files(_seg_bytes()))
    from app.main import registry
    job_obj = registry.get(job["id"])
    target = os.path.join(job_obj.result.output_dir, f"{_anon(SRC1)}.dcm")
    with open(target, "wb") as fh:
        fh.write(b"\x00\x01not dicom anymore")
    r = _seg_coverage(client, job["id"])
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert _find(detail, "source_unreadable")


def test_conflict_source_without_rows_columns(client):
    # 源是带 PixelData 但没有 Rows/Columns 的畸形数据集
    blob_source = _source_bytes(SRC1)
    ds = pydicom.dcmread(io.BytesIO(blob_source))
    del ds.Rows
    del ds.Columns
    files = {"src.dcm": _write_bytes(ds), "seg.dcm": _seg_bytes()}
    detail = _conflict(client, files)
    assert _find(detail, "source_dimension_mismatch")


def test_conflict_multivalue_referenced_segment_number(client):
    # ReferencedSegmentNumber 按标准 VM=1；多值 "1\\2" 无法消歧
    blob = _seg_bytes()
    ds = pydicom.dcmread(io.BytesIO(blob))
    from pydicom.dataelem import DataElement
    sid = ds.PerFrameFunctionalGroupsSequence[0].SegmentIdentificationSequence[0]
    sid[0x0062000B] = DataElement(0x0062000B, "IS", "1\\2")
    detail = _conflict(client, _basic_files(_write_bytes(ds)))
    assert _find(detail, "invalid_segment_number")


def test_union_on_larger_mask(client):
    # 3 帧 16×16，每组前景互不重叠：并集像素数 == 各帧之和
    rows = columns = 16
    ones_a = [(r, c) for r in range(0, 16, 2) for c in (0,)]   # 8 px
    ones_b = [(r, c) for r in range(0, 16, 2) for c in (15,)]  # 8 px
    ones_c = [(0, c) for c in range(1, 15)]                    # 14 px
    seg = _seg_bytes(
        rows=rows,
        columns=columns,
        frames=(
            (1, SRC1, tuple(ones_a)),
            (1, SRC1, tuple(ones_b)),
            (1, SRC1, tuple(ones_c)),
        ),
    )
    sources = {SRC1: {"rows": rows, "columns": columns}}
    job = _upload(client, _basic_files(seg, source_kwargs=sources))
    data = _seg_coverage(client, job["id"]).json()
    item = data["items"][0]
    assert item["frame_count"] == 3
    assert item["image_pixel_count"] == 256
    assert item["foreground_pixel_count"] == 30
    assert item["coverage_ratio"] == pytest.approx(30 / 256)


# --------------------------------------------------------------------------- #
# 安全 / 模式
# --------------------------------------------------------------------------- #
def test_conflict_response_contains_only_anonymized_uids(client):
    """冲突明细不得泄露原始源/SEG UID。"""
    blob = _seg_bytes(frames=((1, "6.999", ((0, 0),)),))
    detail = _conflict(client, _basic_files(blob))
    raw = str(detail)
    assert "6.999" not in raw
    assert SEG_UID not in raw
    assert SRC1 not in raw
    assert _anon("6.999") in raw


def test_endpoint_in_openapi(client):
    r = client.get("/openapi.json")
    paths = r.json()["paths"]
    key = "/jobs/{job_id}/instances/{sop_instance_uid}/seg-coverage"
    assert key in paths
    assert "get" in paths[key]
    schema = paths[key]["get"]
    assert "SegCoverageOut" in str(schema["responses"]["200"])
    endpoints = client.get("/").json()["endpoints"]
    assert "GET /jobs/{job_id}/instances/{sop_instance_uid}/seg-coverage" in endpoints
