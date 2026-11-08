# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Row-level CSV Sources with stable identities and exact record readback."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_IDENTITY_PREFIX = "csv"


class CSVDocumentInput(BaseModel):
    """One offline CSV document plus the identity contract for its rows."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One captured data row keyed by document and primary key value."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A CAPTURED Source holding one exact CSV record with its column order."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def _validated_identity_component(field: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")  # noqa: TRY003
    if value != value.strip():
        raise ValueError(f"{field} must not contain leading or trailing whitespace")  # noqa: TRY003
    return value


def csv_row_source_id(document_id: str, primary_key_value: str, /) -> str:
    """Build the delimiter-safe identity for one CSV row."""

    _validated_identity_component("document_id", document_id)
    if not isinstance(primary_key_value, str) or not primary_key_value.strip():
        raise ValueError("primary_key_value must be a non-blank string")  # noqa: TRY003
    identity = (
        f"{CSV_ROW_IDENTITY_PREFIX}:{quote(document_id, safe='')}:"
        f"{quote(primary_key_value, safe='')}"
    )
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def parse_csv_document(document: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse and validate one whole CSV document into row inputs."""

    document_id = _validated_identity_component("document_id", document.document_id)
    if not isinstance(document.primary_key, str) or not document.primary_key:
        raise ValueError("primary_key must be a non-empty string")  # noqa: TRY003
    try:
        text = document.content.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be strict UTF-8") from error  # noqa: TRY003

    try:
        records = [record for record in csv.reader(io.StringIO(text, newline=""), strict=True)]
    except csv.Error as error:
        raise ValueError("CSV content is malformed") from error  # noqa: TRY003

    if not records or not records[0]:
        raise ValueError("CSV document must contain a non-empty header")  # noqa: TRY003
    columns = tuple(records[0])
    for column in columns:
        if not column.strip():
            raise ValueError("header columns must be non-blank and unique")  # noqa: TRY003
    if len(set(columns)) != len(columns):
        raise ValueError("header columns must be non-blank and unique")  # noqa: TRY003
    if document.primary_key not in columns:
        raise ValueError("primary_key must match one header column exactly")  # noqa: TRY003
    key_index = columns.index(document.primary_key)

    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    for record in records[1:]:
        if not record:
            continue
        if len(record) != len(columns):
            raise ValueError("data records must match the header width")  # noqa: TRY003
        primary_key_value = record[key_index]
        if not primary_key_value.strip():
            raise ValueError("primary key values must be non-blank")  # noqa: TRY003
        if primary_key_value in seen_keys:
            raise ValueError("primary key values must be unique within one document")  # noqa: TRY003
        seen_keys.add(primary_key_value)
        csv_row_source_id(document_id, primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=document.primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


def _row_mapping(source: CSVRowSource, /) -> dict[str, str]:
    return dict(zip(source.columns, source.values, strict=True))


class CSVRowAdapter:
    """Resolve and read row-level CSV Sources."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        if value.primary_key not in value.columns:
            raise ValueError("primary_key must match one column exactly")  # noqa: TRY003
        if len(value.values) != len(value.columns):
            raise ValueError("values must match the column width")  # noqa: TRY003
        return CSVRowSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Project one CSV row as canonical TextEvidence content."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = _row_mapping(source)
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSVRowAdapter(),
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSVDocumentInput",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSVRowAdapter",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
