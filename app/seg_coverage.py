"""匿名 SEG（分割存储）实例的 1-bit 掩码覆盖率统计。

数据来源严格限定为“已完成任务 output 目录中的匿名 DICOM 文件”，
**不重新解压或处理原始 ZIP**。按匿名 SOPInstanceUID 定位任务内**唯一**
成功输出的 SEG 实例（SOP Class = Segmentation Storage），只支持
**未压缩（native）传输语法下的 BINARY 分割（BitsAllocated=1）**。

逐帧关联规则（对应 SEG 的多帧功能组结构）：

* ``PerFrameFunctionalGroupsSequence[i].SegmentIdentificationSequence[0]
  .ReferencedSegmentNumber`` 给出该帧的段号；
* 该帧源图像取自同一逐帧功能组项内的
  ``DerivationImageSequence[*].SourceImageSequence[*].ReferencedSOPInstanceUID``
  （无 DerivationImageSequence 时直接读取功能组项上的
  ``SourceImageSequence``）；这些 UID 已随匿名化在全序列层级一致重写，
  必须能在**本任务的成功输出**中定位到唯一成功输出源图像；
* 同一帧引用多个**不同**源 SOP UID 属于无法消歧的关联，直接报错。

掩码按 DICOM PS3.5 §8.1.1 的 1-bit packed pixel 规则解码：每像素 1 bit，
每行末向**字节**边界补零、每字节最高位（MSB）对应行内最左像素；
全部帧按 (源 UID, 段号) 分组，组内多帧掩码**逐像素取并集**后统计前景
（值 1）像素数，覆盖率 = 前景像素数 / 源图像像素数。结果按
源 UID、段号升序排列，绝不回显原始 UID。

存在以下任一情况时抛出 ``409 SEG_COVERAGE_CONFLICT``，在 ``problems[]``
中一次性给出**含匿名 SOP UID / 帧序号 / 段号**的全部冲突明细，**不给出
任何部分统计**：

* SEG 输出不可读（``unreadable``）、文件内 SOPInstanceUID 与请求不符
  （``sop_instance_uid_mismatch``）、不是 Segmentation Storage
  （``not_segmentation``）；
* 压缩 / Deflated 等非 native 传输语法（``unsupported_transfer_syntax``）、
  ``SegmentationType`` 缺失或不为 BINARY（``unsupported_segmentation_type``）、
  缺 PixelData（``missing_pixel_data``）或像素元数据非法
  （SamplesPerPixel / PhotometricInterpretation / BitsAllocated /
  Rows / Columns / NumberOfFrames / SegmentSequence 等，
  ``invalid_pixel_metadata``）；
* ``PerFrameFunctionalGroupsSequence`` 缺失或帧数不符
  （``frame_functional_groups_missing`` / ``frame_count_mismatch``）、
  PixelData 字节数与声明帧尺寸不符（``invalid_pixel_data_length``）；
* 帧缺段号 / 段号无法解析 / 不在 SegmentSequence 中
  （``missing_segment_number`` / ``invalid_segment_number`` /
  ``unknown_segment_number``）；
* 帧缺源引用 / 同帧引用多个不同源 / 源不在成功输出 / 源输出歧义
  （``missing_source_reference`` / ``ambiguous_source_reference`` /
  ``source_not_in_output`` / ``ambiguous_source_output``）；
* 源图像输出缺失或不可读（``source_output_missing`` /
  ``source_unreadable``）、源 Rows/Columns 缺失或与该 SEG 帧尺寸不一致
  （``source_dimension_mismatch``）。
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ExplicitVRBigEndian,
    ImplicitVRLittleEndian,
    SegmentationStorage,
)

from .anonymizer import FileReport, JobResult

# 仅接受三种未压缩 native 传输语法（与预览接口一致；封装/Deflated 一律拒绝）
UNCOMPRESSED_TRANSFER_SYNTAXES = frozenset(
    {
        str(ImplicitVRLittleEndian),
        str(ExplicitVRLittleEndian),
        str(ExplicitVRBigEndian),
    }
)

# detail.problems[].reason 取值（机器可读）
REASON_UNREADABLE = "unreadable"
REASON_SOP_UID_MISMATCH = "sop_instance_uid_mismatch"
REASON_NOT_SEGMENTATION = "not_segmentation"
REASON_UNSUPPORTED_TRANSFER_SYNTAX = "unsupported_transfer_syntax"
REASON_UNSUPPORTED_SEGMENTATION_TYPE = "unsupported_segmentation_type"
REASON_MISSING_PIXEL_DATA = "missing_pixel_data"
REASON_INVALID_PIXEL_METADATA = "invalid_pixel_metadata"
REASON_FRAME_FG_MISSING = "frame_functional_groups_missing"
REASON_FRAME_COUNT_MISMATCH = "frame_count_mismatch"
REASON_INVALID_PIXEL_DATA_LENGTH = "invalid_pixel_data_length"
REASON_MISSING_SEGMENT_NUMBER = "missing_segment_number"
REASON_INVALID_SEGMENT_NUMBER = "invalid_segment_number"
REASON_UNKNOWN_SEGMENT_NUMBER = "unknown_segment_number"
REASON_MISSING_SOURCE_REFERENCE = "missing_source_reference"
REASON_AMBIGUOUS_SOURCE_REFERENCE = "ambiguous_source_reference"
REASON_SOURCE_NOT_IN_OUTPUT = "source_not_in_output"
REASON_AMBIGUOUS_SOURCE_OUTPUT = "ambiguous_source_output"
REASON_SOURCE_OUTPUT_MISSING = "source_output_missing"
REASON_SOURCE_UNREADABLE = "source_unreadable"
REASON_SOURCE_DIMENSION_MISMATCH = "source_dimension_mismatch"

# 顶层 detail.reason 取值（非 problems 场景）
REASON_AMBIGUOUS_OUTPUT = "ambiguous_output"
REASON_OUTPUT_MISSING = "output_missing"


class SegCoverageError(Exception):
    """SEG 覆盖率统计失败；http_status 由路由层映射为 HTTPException。"""

    def __init__(
        self,
        code: str,
        http_status: int,
        message: str,
        sop_uid: Optional[str] = None,
        reason: Optional[str] = None,
        extra: Optional[dict] = None,
        problems: Optional[list[dict]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        detail: dict = {"code": code, "message": message}
        if sop_uid is not None:
            detail["sop_instance_uid"] = sop_uid
        if reason is not None:
            detail["reason"] = reason
        if extra:
            detail.update(extra)
        if problems:
            detail["problems"] = problems
        self.detail = detail


# --------------------------------------------------------------------------- #
# 输出收集
# --------------------------------------------------------------------------- #
def _successful_reports(result: JobResult) -> list[FileReport]:
    """任务内真正写出匿名文件、且记录了匿名 SOP UID 的报告。"""
    return [
        report
        for report in result.files
        if report.status != "error" and report.output_name and report.sop_instance_uid
    ]


def _locate_unique_output(result: JobResult, sop_uid: str) -> FileReport:
    """按匿名 SOPInstanceUID 定位唯一成功输出；找不到 / 多份输出均明确报错。"""
    matches = [
        report
        for report in _successful_reports(result)
        if report.sop_instance_uid == sop_uid
    ]
    if not matches:
        raise SegCoverageError(
            "INSTANCE_NOT_FOUND",
            404,
            "该匿名 SOPInstanceUID 在本任务的成功输出中不存在",
            sop_uid=sop_uid,
        )
    if len(matches) > 1:
        output_files = sorted(r.output_name for r in matches)
        raise SegCoverageError(
            "AMBIGUOUS_INSTANCE_OUTPUT",
            409,
            "同一匿名 SOPInstanceUID 对应多份输出文件，无法确定要统计的 SEG 实例",
            sop_uid=sop_uid,
            reason=REASON_AMBIGUOUS_OUTPUT,
            extra={"output_files": output_files},
        )
    return matches[0]


# --------------------------------------------------------------------------- #
# 标签取值
# --------------------------------------------------------------------------- #
def _positive_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        n = int(text)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _frame_source_uids(fg_item) -> Optional[list[str]]:
    """逐帧功能组项 -> DerivationImageSequence/SourceImageSequence 内的去重源 UID。

    存在引用序列但其中没有任何非空 ReferencedSOPInstanceUID 时返回空列表；
    两种序列都不存在时返回 None（该帧无源引用）。
    """
    uids: list[str] = []
    found_any_sequence = False
    derivation_seq = getattr(fg_item, "DerivationImageSequence", None)
    if derivation_seq:
        found_any_sequence = True
        for derivation_item in derivation_seq:
            source_seq = getattr(derivation_item, "SourceImageSequence", None)
            if not source_seq:
                continue
            for source_item in source_seq:
                uid = str(
                    getattr(source_item, "ReferencedSOPInstanceUID", "") or ""
                ).strip()
                if uid and uid not in uids:
                    uids.append(uid)
    # 无 DerivationImageSequence 时允许源图像直接挂在逐帧功能组项上
    direct_source_seq = getattr(fg_item, "SourceImageSequence", None)
    if direct_source_seq:
        found_any_sequence = True
        for source_item in direct_source_seq:
            uid = str(
                getattr(source_item, "ReferencedSOPInstanceUID", "") or ""
            ).strip()
            if uid and uid not in uids:
                uids.append(uid)
    return uids if found_any_sequence else None


def _frame_segment_number(fg_item) -> object:
    """读取帧的 ReferencedSegmentNumber。

    返回正整数、None（缺失/为空）或 ``"_invalid"``（存在但无法解析为正整数）。
    VM>1（多值）同样视为非法：一个 SEG 帧只能描绘一个段。
    """
    segment_seq = getattr(fg_item, "SegmentIdentificationSequence", None)
    if not segment_seq:
        return None
    raw = getattr(segment_seq[0], "ReferencedSegmentNumber", None)
    if raw is None or not str(raw).strip():
        return None
    if isinstance(raw, pydicom.multival.MultiValue):
        values = list(raw)
        if len(values) != 1:
            return "_invalid"
        raw = values[0]
    elif isinstance(raw, (list, tuple)):
        if len(raw) != 1:
            return "_invalid"
        raw = raw[0]
    text = str(raw).strip()
    try:
        n = int(text)
    except (TypeError, ValueError):
        return "_invalid"
    return n if n > 0 else "_invalid"


# --------------------------------------------------------------------------- #
# 1-bit 掩码解码（DICOM PS3.5 §8.1.1：MSB-first，行末补零到字节边界）
# --------------------------------------------------------------------------- #
def _unpack_1bit_frames(
    pixel_data: bytes, frames: int, rows: int, columns: int
) -> np.ndarray:
    """把 native 1-bit packed PixelData 解码为 (frames, rows, columns) 的 0/1 数组。"""
    bytes_per_row = (columns + 7) // 8
    frame_stride = rows * bytes_per_row
    packed = np.frombuffer(pixel_data, dtype=np.uint8, count=frame_stride * frames)
    packed = packed.reshape(frames, rows, bytes_per_row)
    bits = np.unpackbits(packed, axis=2, bitorder="big")
    return bits[:, :, :columns].astype(np.uint8)


# --------------------------------------------------------------------------- #
# 源图像索引
# --------------------------------------------------------------------------- #
def _build_source_index(reports: list[FileReport]) -> dict[str, list[FileReport]]:
    """匿名 SOPInstanceUID -> 成功输出报告列表（多于 1 个即源输出歧义）。"""
    by_sop: dict[str, list[FileReport]] = {}
    for report in reports:
        by_sop.setdefault(report.sop_instance_uid, []).append(report)
    return by_sop


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def analyze_seg_coverage(job, sop_uid: str) -> dict:
    """定位任务唯一 SEG 成功输出并统计逐 (源 UID, 段号) 的 1-bit 掩码覆盖率。

    任何定位/解析/关联/尺寸错误均抛 SegCoverageError，且不返回部分统计。
    """
    result: JobResult = job.result
    all_reports = _successful_reports(result)
    source_index = _build_source_index(all_reports)

    report = _locate_unique_output(result, sop_uid)
    path = os.path.join(result.output_dir, report.output_name)
    if not os.path.isfile(path):
        raise SegCoverageError(
            "INSTANCE_OUTPUT_MISSING",
            404,
            "匿名输出文件在任务目录中不存在（可能已被清理）",
            sop_uid=sop_uid,
            reason=REASON_OUTPUT_MISSING,
            extra={"output_file": report.output_name},
        )

    problems: list[dict] = []

    def add_problem(reason: str, message: str, **fields: object) -> None:
        item = {
            "sop_instance_uid": sop_uid,
            "output_file": report.output_name,
            "reason": reason,
            "message": message,
        }
        item.update(fields)
        problems.append(item)

    # ---- 读取 SEG ------------------------------------------------------- #
    try:
        ds = pydicom.dcmread(path, force=False)
    except (InvalidDicomError, OSError, EOFError) as exc:
        raise SegCoverageError(
            "SEG_COVERAGE_CONFLICT",
            409,
            "匿名 SEG 输出不是可读取的 DICOM 文件，无法统计覆盖率",
            sop_uid=sop_uid,
            extra={"conflict_count": 1},
            problems=[{
                "sop_instance_uid": sop_uid,
                "output_file": report.output_name,
                "reason": REASON_UNREADABLE,
                "message": f"匿名输出不是可读取的 DICOM 文件：{exc}",
            }],
        ) from exc
    except Exception as exc:  # 标签编码损坏等
        raise SegCoverageError(
            "SEG_COVERAGE_CONFLICT",
            409,
            "读取匿名 SEG 输出失败，无法统计覆盖率",
            sop_uid=sop_uid,
            extra={"conflict_count": 1},
            problems=[{
                "sop_instance_uid": sop_uid,
                "output_file": report.output_name,
                "reason": REASON_UNREADABLE,
                "message": f"读取匿名输出失败：{exc}",
            }],
        ) from exc

    actual_sop = str(getattr(ds, "SOPInstanceUID", "") or "").strip()
    if actual_sop and actual_sop != sop_uid:
        add_problem(
            REASON_SOP_UID_MISMATCH,
            "文件内 SOPInstanceUID 与请求的匿名 SOP UID 不一致，拒绝统计："
            f"文件内为 {actual_sop}，请求为 {sop_uid}",
        )

    sop_class = str(getattr(ds, "SOPClassUID", "") or "").strip()
    if sop_class != str(SegmentationStorage):
        add_problem(
            REASON_NOT_SEGMENTATION,
            "该实例不是 Segmentation Storage（SOPClassUID="
            f"{sop_class or '<缺失>'}），无法统计分割覆盖率",
            sop_class_uid=sop_class,
        )

    ts_uid = str(
        getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "") or ""
    ).strip()
    if ts_uid not in UNCOMPRESSED_TRANSFER_SYNTAXES:
        add_problem(
            REASON_UNSUPPORTED_TRANSFER_SYNTAX,
            "覆盖率统计仅支持未压缩传输语法（Implicit/Explicit VR Little Endian、"
            f"Explicit VR Big Endian），收到 TransferSyntaxUID={ts_uid or '<缺失>'}",
            transfer_syntax_uid=ts_uid,
        )

    segmentation_type = str(getattr(ds, "SegmentationType", "") or "").strip().upper()
    if segmentation_type != "BINARY":
        add_problem(
            REASON_UNSUPPORTED_SEGMENTATION_TYPE,
            "覆盖率统计仅支持 SegmentationType=BINARY 的未压缩 1-bit 分割，收到 "
            f"SegmentationType={segmentation_type or '<缺失>'}",
            segmentation_type=segmentation_type or None,
        )

    # ---- 像素元数据 ----------------------------------------------------- #
    rows = _positive_int(getattr(ds, "Rows", None))
    columns = _positive_int(getattr(ds, "Columns", None))
    if rows is None or columns is None:
        add_problem(
            REASON_INVALID_PIXEL_METADATA,
            "SEG 缺少有效的 Rows/Columns，无法解码掩码",
        )

    try:
        samples = int(getattr(ds, "SamplesPerPixel"))
    except (TypeError, ValueError):
        samples = None
    if samples != 1:
        add_problem(
            REASON_INVALID_PIXEL_METADATA,
            f"覆盖率统计仅支持 SamplesPerPixel=1 的掩码，收到 {samples!r}",
        )

    photometric = str(
        getattr(ds, "PhotometricInterpretation", "") or ""
    ).strip().upper()
    if photometric != "MONOCHROME2":
        add_problem(
            REASON_INVALID_PIXEL_METADATA,
            "覆盖率统计仅支持 PhotometricInterpretation=MONOCHROME2 的掩码，收到 "
            f"{photometric or '<缺失>'}",
        )

    try:
        bits_allocated = int(getattr(ds, "BitsAllocated"))
    except (TypeError, ValueError):
        bits_allocated = None
    if bits_allocated != 1:
        add_problem(
            REASON_INVALID_PIXEL_METADATA,
            f"覆盖率统计仅支持 BitsAllocated=1 的未压缩 BINARY 分割，收到 "
            f"{bits_allocated!r}",
        )

    try:
        pixel_repr = int(getattr(ds, "PixelRepresentation"))
    except (TypeError, ValueError):
        pixel_repr = None
    if pixel_repr != 0:
        add_problem(
            REASON_INVALID_PIXEL_METADATA,
            f"1-bit BINARY 掩码的 PixelRepresentation 必须为 0（无符号），收到 "
            f"{pixel_repr!r}",
        )

    raw_nof = getattr(ds, "NumberOfFrames", None)
    declared = _positive_int(raw_nof)
    if declared is None:
        add_problem(
            REASON_INVALID_PIXEL_METADATA,
            "SEG 缺少有效的正整数 NumberOfFrames，无法逐帧解码掩码",
            declared_number_of_frames=(
                "" if raw_nof is None else str(raw_nof)
            ),
        )
        declared = 0  # 后续不再索引逐帧序列

    segment_seq = getattr(ds, "SegmentSequence", None)
    known_segment_numbers: set[int] = set()
    if not segment_seq:
        add_problem(
            REASON_INVALID_PIXEL_METADATA,
            "SEG 缺少 SegmentSequence，无法校验帧段号",
        )
    else:
        for index, segment_item in enumerate(segment_seq):
            number = _positive_int(getattr(segment_item, "SegmentNumber", None))
            if number is None:
                add_problem(
                    REASON_INVALID_PIXEL_METADATA,
                    f"SegmentSequence[{index}] 缺少有效的正整数 SegmentNumber",
                    segment_index=index,
                )
                continue
            if number in known_segment_numbers:
                add_problem(
                    REASON_INVALID_PIXEL_METADATA,
                    f"SegmentSequence 中段号 {number} 重复定义",
                    segment_number=number,
                )
            known_segment_numbers.add(number)

    has_pixel_data = "PixelData" in ds and bool(ds.PixelData)
    if not has_pixel_data:
        add_problem(
            REASON_MISSING_PIXEL_DATA,
            "SEG 不包含非空 PixelData，无法解码掩码",
        )

    per_frame_seq = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if per_frame_seq is None:
        add_problem(
            REASON_FRAME_FG_MISSING,
            "SEG 缺少 PerFrameFunctionalGroupsSequence，无法逐帧关联段号与源图像",
        )
    elif declared and len(per_frame_seq) != declared:
        add_problem(
            REASON_FRAME_COUNT_MISMATCH,
            f"NumberOfFrames 声明 {declared} 帧，但 PerFrameFunctionalGroupsSequence "
            f"含 {len(per_frame_seq)} 个条目，无法逐帧对应",
            declared_number_of_frames=str(declared),
            per_frame_item_count=len(per_frame_seq),
        )

    # ---- 像素字节长度 --------------------------------------------------- #
    pixel_length_ok = False
    if has_pixel_data and rows and columns and declared:
        bytes_per_row = (columns + 7) // 8
        expected = rows * bytes_per_row * declared
        actual_length = len(ds.PixelData)
        # PixelData 作为 DICOM 元素会向偶数字节补齐，允许正好多 1 个补零字节
        if actual_length not in (expected, expected + (expected & 1)):
            add_problem(
                REASON_INVALID_PIXEL_DATA_LENGTH,
                f"PixelData 长度 {actual_length} 字节与声明的 "
                f"{declared} 帧 × {rows} 行 × {columns} 列（{expected} 字节"
                f"{'+ 偶数字节补齐' if expected & 1 else ''}）不符",
                pixel_data_length=actual_length,
                expected_pixel_data_length=expected,
            )
        else:
            pixel_length_ok = True

    # 结构性元数据异常时不再继续逐帧关联/解码（帧信息本身已不可信）
    if problems:
        raise SegCoverageError(
            "SEG_COVERAGE_CONFLICT",
            409,
            "SEG 实例存在结构、传输语法、像素元数据或帧数异常，"
            "无法给出完整的分割覆盖率（不输出部分统计）",
            sop_uid=sop_uid,
            extra={"conflict_count": len(problems)},
            problems=sorted(
                problems,
                key=lambda p: (
                    p.get("frame_index") is not None,
                    p.get("frame_index") if p.get("frame_index") is not None else -1,
                    p["reason"],
                ),
            ),
        )

    # ---- 逐帧关联（段号 + 源 UID）-------------------------------------- #
    frame_bindings: list[tuple[int, int, str]] = []  # (frame_index, segment, src uid)
    for frame_index, fg_item in enumerate(per_frame_seq):
        segment_number = _frame_segment_number(fg_item)
        if segment_number is None:
            add_problem(
                REASON_MISSING_SEGMENT_NUMBER,
                f"第 {frame_index} 帧缺少 SegmentIdentificationSequence"
                ".ReferencedSegmentNumber",
                frame_index=frame_index,
            )
        elif segment_number == "_invalid":
            add_problem(
                REASON_INVALID_SEGMENT_NUMBER,
                f"第 {frame_index} 帧的 ReferencedSegmentNumber 无法解析为"
                "唯一正整数段号",
                frame_index=frame_index,
            )
        elif segment_number not in known_segment_numbers:
            add_problem(
                REASON_UNKNOWN_SEGMENT_NUMBER,
                f"第 {frame_index} 帧引用的段号 {segment_number} 不在 "
                "SegmentSequence 中",
                frame_index=frame_index,
                segment_number=segment_number,

            )

        source_uids = _frame_source_uids(fg_item)
        source_uid: Optional[str] = None
        if source_uids is None:
            add_problem(
                REASON_MISSING_SOURCE_REFERENCE,
                f"第 {frame_index} 帧的功能组内没有 "
                "DerivationImageSequence/SourceImageSequence 源图像引用",
                frame_index=frame_index,
                segment_number=(
                    segment_number if isinstance(segment_number, int) else None
                ),
            )
        elif not source_uids:
            add_problem(
                REASON_MISSING_SOURCE_REFERENCE,
                f"第 {frame_index} 帧的源图像引用序列中没有任何 "
                "ReferencedSOPInstanceUID",
                frame_index=frame_index,
                segment_number=(
                    segment_number if isinstance(segment_number, int) else None
                ),
            )
        elif len(source_uids) > 1:
            add_problem(
                REASON_AMBIGUOUS_SOURCE_REFERENCE,
                f"第 {frame_index} 帧同时引用 {len(source_uids)} 个不同的源 "
                "SOPInstanceUID，无法唯一关联",
                frame_index=frame_index,
                segment_number=(
                    segment_number if isinstance(segment_number, int) else None
                ),
                referenced_sop_instance_uids=sorted(source_uids),
            )
        else:
            source_uid = source_uids[0]
            matches = source_index.get(source_uid, [])
            if not matches:
                add_problem(
                    REASON_SOURCE_NOT_IN_OUTPUT,
                    f"第 {frame_index} 帧引用的源 SOPInstanceUID "
                    f"{source_uid} 不在本任务的成功输出中",
                    frame_index=frame_index,
                    segment_number=(
                        segment_number if isinstance(segment_number, int) else None
                    ),
                    source_sop_instance_uid=source_uid,
                )
            elif len(matches) > 1:
                add_problem(
                    REASON_AMBIGUOUS_SOURCE_OUTPUT,
                    f"第 {frame_index} 帧引用的源 SOPInstanceUID {source_uid} "
                    "对应多份输出文件，无法确定唯一源图像",
                    frame_index=frame_index,
                    segment_number=(
                        segment_number if isinstance(segment_number, int) else None
                    ),
                    source_sop_instance_uid=source_uid,
                    output_files=sorted(r.output_name for r in matches),
                )

        if isinstance(segment_number, int) and source_uid is not None:
            frame_bindings.append((frame_index, segment_number, source_uid))

    if problems:
        raise SegCoverageError(
            "SEG_COVERAGE_CONFLICT",
            409,
            "SEG 逐帧段号或源图像引用缺失、无法解析或存在歧义，"
            "无法给出完整的分割覆盖率（不输出部分统计）",
            sop_uid=sop_uid,
            extra={"conflict_count": len(problems)},
            problems=sorted(
                problems,
                key=lambda p: (
                    p.get("frame_index", -1),
                    p["reason"],
                ),
            ),
        )

    # ---- 解码掩码（长度已校验，此处不应失败；失败按像素数据异常处理）---- #
    try:
        masks = _unpack_1bit_frames(
            bytes(ds.PixelData), declared, rows, columns
        )
    except Exception as exc:
        raise SegCoverageError(
            "SEG_COVERAGE_CONFLICT",
            409,
            "1-bit 掩码像素数据无法解码，无法统计覆盖率",
            sop_uid=sop_uid,
            extra={"conflict_count": 1},
            problems=[{
                "sop_instance_uid": sop_uid,
                "output_file": report.output_name,
                "reason": REASON_INVALID_PIXEL_METADATA,
                "message": f"1-bit 像素数据解码失败：{exc}",
            }],
        ) from exc

    # ---- 读取源图像尺寸并逐组取并集 ------------------------------------ #
    # 同一源 SOP UID 的全部帧必与同一物理源图像（尺寸一致）；源只需读取一次
    union_by_key: dict[tuple[str, int], np.ndarray] = {}
    frame_counts: dict[tuple[str, int], int] = {}
    source_dimensions: dict[str, tuple[int, int]] = {}
    source_reports_needed = {
        source_uid: source_index[source_uid][0]
        for _, _, source_uid in frame_bindings
    }

    for source_uid, source_report in source_reports_needed.items():
        source_path = os.path.join(result.output_dir, source_report.output_name)
        if not os.path.isfile(source_path):
            for frame_index, _, bound_uid in frame_bindings:
                if bound_uid != source_uid:
                    continue
                add_problem(
                    REASON_SOURCE_OUTPUT_MISSING,
                    f"源 SOPInstanceUID {source_uid} 的匿名输出文件在任务目录中"
                    "不存在（可能已被清理）",
                    frame_index=frame_index,
                    source_sop_instance_uid=source_uid,
                    output_file=source_report.output_name,
                )
            continue
        try:
            source_ds = pydicom.dcmread(source_path, force=False, stop_before_pixels=True)
        except (InvalidDicomError, OSError, EOFError) as exc:
            for frame_index, _, bound_uid in frame_bindings:
                if bound_uid != source_uid:
                    continue
                add_problem(
                    REASON_SOURCE_UNREADABLE,
                    f"源 SOPInstanceUID {source_uid} 的匿名输出不可读：{exc}",
                    frame_index=frame_index,
                    source_sop_instance_uid=source_uid,
                    output_file=source_report.output_name,
                )
            continue
        except Exception as exc:  # 标签编码损坏等
            for frame_index, _, bound_uid in frame_bindings:
                if bound_uid != source_uid:
                    continue
                add_problem(
                    REASON_SOURCE_UNREADABLE,
                    f"读取源 SOPInstanceUID {source_uid} 的匿名输出失败：{exc}",
                    frame_index=frame_index,
                    source_sop_instance_uid=source_uid,
                    output_file=source_report.output_name,
                )
            continue

        src_rows = _positive_int(getattr(source_ds, "Rows", None))
        src_cols = _positive_int(getattr(source_ds, "Columns", None))
        if src_rows is None or src_cols is None:
            for frame_index, _, bound_uid in frame_bindings:
                if bound_uid != source_uid:
                    continue
                add_problem(
                    REASON_SOURCE_DIMENSION_MISMATCH,
                    f"源 SOPInstanceUID {source_uid} 缺少有效的 Rows/Columns，"
                    "无法确定图像像素数",
                    frame_index=frame_index,
                    source_sop_instance_uid=source_uid,
                    output_file=source_report.output_name,
                )
            continue
        source_dimensions[source_uid] = (src_rows, src_cols)
        if src_rows != rows or src_cols != columns:
            for frame_index, _, bound_uid in frame_bindings:
                if bound_uid != source_uid:
                    continue
                add_problem(
                    REASON_SOURCE_DIMENSION_MISMATCH,
                    f"第 {frame_index} 帧的源 SOPInstanceUID {source_uid} 尺寸为 "
                    f"{src_rows}×{src_cols}，与 SEG 帧尺寸 {rows}×{columns} 不一致",
                    frame_index=frame_index,
                    source_sop_instance_uid=source_uid,
                    output_file=source_report.output_name,
                    source_rows=src_rows,
                    source_columns=src_cols,
                    seg_rows=rows,
                    seg_columns=columns,
                )

    if problems:
        raise SegCoverageError(
            "SEG_COVERAGE_CONFLICT",
            409,
            "SEG 引用的源图像缺失、不可读或尺寸与掩码不一致，"
            "无法给出完整的分割覆盖率（不输出部分统计）",
            sop_uid=sop_uid,
            extra={"conflict_count": len(problems)},
            problems=sorted(
                problems,
                key=lambda p: (
                    p.get("frame_index", -1),
                    p["reason"],
                ),
            ),
        )

    for frame_index, segment_number, source_uid in frame_bindings:
        key = (source_uid, segment_number)
        frame_mask = masks[frame_index]
        if key not in union_by_key:
            union_by_key[key] = frame_mask.copy()
            frame_counts[key] = 1
        else:
            np.logical_or(union_by_key[key], frame_mask, out=union_by_key[key])
            frame_counts[key] += 1

    items: list[dict] = []
    for source_uid, segment_number in sorted(union_by_key):
        union = union_by_key[(source_uid, segment_number)]
        src_rows, src_cols = source_dimensions[source_uid]
        image_pixels = src_rows * src_cols
        foreground = int(int(union.sum()))
        items.append({
            "source_sop_instance_uid": source_uid,
            "segment_number": segment_number,
            "frame_count": frame_counts[(source_uid, segment_number)],
            "foreground_pixel_count": foreground,
            "image_pixel_count": image_pixels,
            "coverage_ratio": foreground / image_pixels,
        })

    return {
        "sop_instance_uid": sop_uid,
        "output_file": report.output_name,
        "rows": rows,
        "columns": columns,
        "number_of_frames": declared,
        "item_count": len(items),
        "items": items,
    }
