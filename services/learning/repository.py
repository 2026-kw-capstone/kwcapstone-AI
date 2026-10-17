"""PostgreSQL is authoritative; vector results NEVER feed aggregate statistics."""
import json
import math
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Protocol
from .models import Observation, LearningConfig, learner_key, POLICY_VERSION, utc_now


class LearningRepository(Protocol):
    def get_state(self, key: str) -> dict: ...
    def observations(self, key: str) -> list: ...
    def append(self, observation: Observation) -> dict: ...
    def rebuild(self, key: str) -> dict: ...
    def record_design(self, key: str, plan: dict, telemetry: dict) -> None: ...
    def search_feedback(self, key: str, vector: list, targets: list, error_types: list, category: str) -> list: ...
    def attach_embeddings(self, key: str, record_id: str, embed: object) -> None: ...
    def update_telemetry(self, key: str, record_id: str, telemetry: dict) -> None: ...


def choose_priorities(state, config, now=None):
    """Explainable policy over FULL SQL aggregates, never over retrieval hits."""
    now = datetime.fromisoformat(now or utc_now())
    state["review"], state["priorities"] = [], []
    for target, t in state["targets"].items():
        t["repeated_error"] = t["recent_failures"] >= config.repeat_errors
        due = datetime.fromisoformat(t["last_practice"]) + timedelta(days=config.review_days)
        t["review_at"] = due.isoformat()
        if t["attempts"] < config.min_attempts:
            continue
        if t["repeated_error"]:
            state["priorities"].append({"target": target, "difficulty": "easy", "reason": "recent_repeated_error"})
        elif not t["recent"][0]["errors"] and due <= now:
            item = {"target": target, "difficulty": "review", "reason": "spaced_success_review"}
            state["review"].append(item)
            state["priorities"].append(item)
        elif not t["recent"][0]["errors"]:
            state["priorities"].append({"target": target, "difficulty": "transfer", "reason": "check_new_words_and_context"})
    state["priorities"].sort(key=lambda x: (x["difficulty"] != "easy", x["difficulty"] != "review",
                                           -state["targets"][x["target"]]["error_rate"], x["target"]))
    return state


# All qualified observations of the learner are the population. The bounded window
# below affects recent trend only; totals are grouped before any window filtering.
STATS_SQL = """
WITH eligible AS (
 SELECT t.*, o.sequence, o.processed_at, o.reference_text, o.context, o.versions,
        row_number() OVER (PARTITION BY t.target ORDER BY o.sequence DESC) AS rn
 FROM learning_targets t JOIN learning_observations o USING(learner_key,record_id)
 WHERE t.learner_key=%s AND o.quality_status='sufficient'
), totals AS (
 SELECT target, sum(attempts)::bigint AS attempts, sum(errors)::bigint AS errors,
        sum(errors)::float / sum(attempts) AS error_rate, max(processed_at) AS last_practice
 FROM eligible GROUP BY target
), recent AS (
 SELECT target, count(*) FILTER(WHERE errors>0) AS recent_failures,
        count(*) AS recent_count,
        avg(errors::float/attempts) FILTER(WHERE rn<=%s) AS after_rate,
        avg(errors::float/attempts) FILTER(WHERE rn>%s) AS before_rate,
        jsonb_agg(jsonb_build_object('attempts',attempts,'errors',errors,'time',processed_at,
                 'record_id',record_id,'reference',reference_text,'context',context,'versions',versions)
                 ORDER BY sequence DESC) AS recent
 FROM eligible WHERE rn<=%s GROUP BY target
)
SELECT totals.*, recent.recent_failures, recent.recent,
 CASE WHEN recent_count<4 OR before_rate IS NULL THEN 'insufficient'
      WHEN after_rate<before_rate THEN 'improving'
      WHEN after_rate>before_rate THEN 'worsening' ELSE 'stable' END AS trend
FROM totals JOIN recent USING(target)
"""


def vector_literal(vector):
    if len(vector) != 1536 or any(not math.isfinite(float(x)) for x in vector) or not any(float(x) for x in vector):
        raise ValueError("Expected a nonzero vector of 1536 finite embedding components")
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


class PostgresLearningRepository:
    def __init__(self, dsn=None, config=None, connect=None):
        self.config = config or LearningConfig()
        self.dsn = dsn or self.config.database_url
        if not self.dsn:
            raise ValueError("LEARNING_DATABASE_URL is required for personalized persistence")
        if connect is None:
            import psycopg
            connect = psycopg.connect
        self._connect = connect

    @contextmanager
    def connection(self):
        from psycopg.rows import dict_row
        # No global connection shared across FastAPI worker threads; connections close
        # even on errors. Explicit connect/query/lock deadlines bound DB work.
        with self._connect(self.dsn, autocommit=True, connect_timeout=3, row_factory=dict_row,
                           options="-c statement_timeout=4000 -c lock_timeout=3000") as conn:
            yield conn

    def _key(self, key):
        key = learner_key(key)
        if not key:
            raise ValueError("Anonymous history is disabled")
        return key

    def _state(self, conn, key):
        window = self.config.recent_window
        rows = conn.execute(STATS_SQL, (key, max(1, window//2), max(1, window//2), window)).fetchall()
        state = {"targets": {}, "policy_version": POLICY_VERSION}
        for row in rows:
            row = dict(row)
            target = row.pop("target")
            row["last_practice"] = row["last_practice"].isoformat()
            state["targets"][target] = row
        meta = conn.execute("""SELECT count(*) AS observations,
                    count(*) FILTER(WHERE quality_status<>'sufficient') AS excluded_observations,
                    max(processed_at) AS last_practice FROM learning_observations WHERE learner_key=%s""", (key,)).fetchone()
        state.update(meta)
        if state["last_practice"]:
            state["last_practice"] = state["last_practice"].isoformat()
        state["error_patterns"] = conn.execute("""SELECT e.error_type, count(*) AS count
            FROM learning_errors e JOIN learning_observations o USING(learner_key,record_id)
            WHERE e.learner_key=%s AND o.quality_status='sufficient' GROUP BY e.error_type""", (key,)).fetchall()
        return choose_priorities(state, self.config)

    def get_state(self, key):
        key = self._key(key)
        with self.connection() as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            return self._state(conn, key)

    def observations(self, key):
        key = self._key(key)
        with self.connection() as conn:
            return [Observation.model_validate(r["payload"]) for r in conn.execute(
                "SELECT payload FROM learning_observations WHERE learner_key=%s ORDER BY sequence", (key,))]

    def _lock(self, conn, key):
        # Serialization is per learner; collisions only reduce parallelism, never mix users.
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (key,))

    def _write_state(self, conn, key):
        from psycopg.types.json import Jsonb
        state = self._state(conn, key)
        conn.execute("""INSERT INTO learning_states(learner_key,policy_version,payload) VALUES(%s,%s,%s)
                    ON CONFLICT(learner_key) DO UPDATE SET payload=excluded.payload,
                    policy_version=excluded.policy_version, updated_at=now()""", (key, POLICY_VERSION, Jsonb(state)))
        return state

    def append(self, observation):
        from psycopg.types.json import Jsonb
        o = observation
        key = self._key(o.learner_key)
        with self.connection() as conn, conn.transaction():
            self._lock(conn, key)
            conn.execute("""INSERT INTO learning_observations
                (record_id,learner_key,processed_at,context,reference_text,stt_text,acoustic_text,method,fallback,
                 pronunciation_score,meaning_score,voice,quality_status,quality_reasons,versions,telemetry,payload)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                         (o.record_id,key,o.processed_at,Jsonb(o.context),o.reference_text,o.stt_text,o.acoustic_text,
                          o.method,o.fallback,o.pronunciation_score,o.meaning_score,Jsonb(o.voice),o.quality.status,
                          Jsonb(o.quality.reasons),Jsonb(o.versions),Jsonb(o.telemetry),Jsonb(o.model_dump())))
            for target, count in o.targets.items():
                conn.execute("INSERT INTO learning_targets VALUES(%s,%s,%s,%s,%s)",
                             (key,o.record_id,target,count["attempts"],count["errors"]))
            for word in o.words:
                if word["errorType"] != "equal":
                    conn.execute("""INSERT INTO learning_errors(learner_key,record_id,ref_index,hyp_index,error_type,detail)
                                    VALUES(%s,%s,%s,%s,%s,%s)""",
                                 (key,o.record_id,word["refIndex"],word.get("hypIndex"),word["errorType"],Jsonb(word)))
            for insertion in o.insertions:
                conn.execute("INSERT INTO learning_errors(learner_key,record_id,hyp_index,error_type,detail) VALUES(%s,%s,%s,'insert',%s)",
                             (key,o.record_id,insertion.get("hypIndex"),Jsonb(insertion)))
            for category, field in (("meaning","meaningDeliveryFeedback"),("pronunciation","pronunciationFeedback")):
                conn.execute("""INSERT INTO learning_feedback(feedback_id,learner_key,record_id,category,text,evidence)
                                VALUES(%s,%s,%s,%s,%s,%s)""", (str(uuid.uuid4()),key,o.record_id,category,o.feedback[field],
                                Jsonb({"quality":o.quality.model_dump(),"sources":o.telemetry.get("coach",{}),"versions":o.versions})))
            return self._write_state(conn, key)

    def rebuild(self, key):
        key = self._key(key)
        with self.connection() as conn, conn.transaction():
            self._lock(conn, key)
            return self._write_state(conn, key)

    def record_design(self, key, plan, telemetry):
        from psycopg.types.json import Jsonb
        with self.connection() as conn:
            conn.execute("INSERT INTO learning_design_events(learner_key,plan,telemetry) VALUES(%s,%s,%s)",
                         (self._key(key),Jsonb(plan),Jsonb(telemetry)))

    def attach_embeddings(self, key, record_id, embed):
        """Optional enrichment AFTER the atomic core write; durable pending/failed state."""
        key = self._key(key)
        with self.connection() as conn:
            rows = conn.execute("""SELECT f.feedback_id, f.text, o.context, o.reference_text
                    FROM learning_feedback f JOIN learning_observations o USING(learner_key,record_id)
                    WHERE f.learner_key=%s AND f.record_id=%s""", (key,record_id)).fetchall()
        if not rows:
            return
        texts = []
        for row in rows:
            texts += [row["text"], json.dumps(row["context"], ensure_ascii=False)]
        try:
            vectors = embed(texts)
            if len(vectors) != len(texts):
                raise ValueError("embedding_count")
            values = [vector_literal(v) for v in vectors]
            with self.connection() as conn, conn.transaction():
                for i,row in enumerate(rows):
                    for j,kind in enumerate(("feedback","context")):
                        conn.execute("""INSERT INTO learning_embeddings VALUES(%s,%s,%s,%s,%s::vector)
                            ON CONFLICT(learner_key,feedback_id,kind) DO UPDATE SET embedding=excluded.embedding,model=excluded.model""",
                                     (key,row["feedback_id"],kind,self.config.embedding_model,values[2*i+j]))
                    conn.execute("UPDATE learning_feedback SET embedding_status='ready' WHERE learner_key=%s AND feedback_id=%s", (key,row["feedback_id"]))
        except Exception:
            with self.connection() as conn:
                conn.execute("UPDATE learning_feedback SET embedding_status='failed' WHERE learner_key=%s AND record_id=%s", (key,record_id))
            raise

    def search_feedback(self, key, vector, targets, error_types, category="pronunciation"):
        key = self._key(key)
        if not targets or category not in {"pronunciation","meaning"}:
            return []
        # Exact cosine search AFTER learner + structured restrictions. Similarity is
        # a retrieval rank, not a calibrated probability or measurement confidence.
        with self.connection() as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            rows = conn.execute("""SELECT f.feedback_id, f.record_id, f.text, f.category, o.sequence,
                    o.quality_status, o.quality_reasons, o.versions, o.context,
                    o.payload AS observation, 1-min(e.embedding <=> %s::vector) AS similarity
                FROM learning_feedback f JOIN learning_observations o USING(learner_key,record_id)
                JOIN learning_embeddings e USING(learner_key,feedback_id)
                WHERE f.learner_key=%s AND f.category=%s AND e.model=%s
                AND EXISTS (SELECT 1 FROM learning_targets t WHERE t.learner_key=f.learner_key
                            AND t.record_id=f.record_id AND t.target=ANY(%s))
                AND (%s::text[]='{}' OR EXISTS (SELECT 1 FROM learning_errors er WHERE er.learner_key=f.learner_key
                            AND er.record_id=f.record_id AND er.error_type=ANY(%s)))
                GROUP BY f.feedback_id,f.record_id,f.text,f.category,o.sequence,o.quality_status,o.quality_reasons,
                         o.versions,o.context,o.payload
                HAVING 1-min(e.embedding <=> %s::vector)>=%s
                ORDER BY similarity DESC,o.sequence DESC LIMIT %s""",
                (vector_literal(vector),key,category,self.config.embedding_model,targets,error_types,error_types,
                 vector_literal(vector),self.config.similarity_min,self.config.search_limit)).fetchall()
            for row in rows:
                row["feedback_id"], row["record_id"] = str(row["feedback_id"]), str(row["record_id"])
                row["followups"] = conn.execute("""SELECT o.record_id, o.quality_status, o.versions,
                        o.pronunciation_score, o.meaning_score, t.target,t.attempts,t.errors,o.processed_at
                    FROM learning_observations o JOIN learning_targets t USING(learner_key,record_id)
                    WHERE o.learner_key=%s AND o.sequence>%s AND t.target=ANY(%s)
                    ORDER BY o.sequence DESC LIMIT 20""", (key,row["sequence"],targets)).fetchall()
                for follow in row["followups"]:
                    follow["record_id"] = str(follow["record_id"])
                    follow["processed_at"] = follow["processed_at"].isoformat()
            return rows

    def update_telemetry(self, key, record_id, telemetry):
        from psycopg.types.json import Jsonb
        with self.connection() as conn:
            conn.execute("""UPDATE learning_observations SET telemetry=%s,
                payload=jsonb_set(payload,'{telemetry}',%s) WHERE learner_key=%s AND record_id=%s""",
                         (Jsonb(telemetry),Jsonb(telemetry),self._key(key),record_id))
