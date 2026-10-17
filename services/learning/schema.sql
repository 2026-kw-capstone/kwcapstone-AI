-- Apply with: psql "$LEARNING_DATABASE_URL" -v ON_ERROR_STOP=1 -f services/learning/schema.sql
-- Extension installation is an explicit DBA/migration action, never done by a request.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS learning_observations (
    sequence BIGSERIAL PRIMARY KEY,
    record_id UUID NOT NULL UNIQUE,
    learner_key TEXT NOT NULL CHECK (btrim(learner_key) <> ''),
    processed_at TIMESTAMPTZ NOT NULL,
    context JSONB NOT NULL,
    reference_text TEXT NOT NULL,
    stt_text TEXT NOT NULL,
    acoustic_text TEXT,
    method TEXT NOT NULL,
    fallback BOOLEAN NOT NULL,
    pronunciation_score DOUBLE PRECISION NOT NULL,
    meaning_score INTEGER NOT NULL,
    voice JSONB NOT NULL,
    quality_status TEXT NOT NULL CHECK (quality_status IN ('sufficient','needs_review','unmeasurable')),
    quality_reasons JSONB NOT NULL,
    versions JSONB NOT NULL,
    telemetry JSONB NOT NULL,
    payload JSONB NOT NULL,
    UNIQUE (learner_key, record_id)
);
CREATE INDEX IF NOT EXISTS learning_observations_learner ON learning_observations(learner_key, sequence);
CREATE TABLE IF NOT EXISTS learning_targets (
    learner_key TEXT NOT NULL,
    record_id UUID NOT NULL,
    target TEXT NOT NULL,
    attempts INTEGER NOT NULL CHECK (attempts > 0),
    errors INTEGER NOT NULL CHECK (errors >= 0 AND errors <= attempts),
    PRIMARY KEY (learner_key, record_id, target),
    FOREIGN KEY (learner_key, record_id) REFERENCES learning_observations(learner_key, record_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS learning_errors (
    id BIGSERIAL PRIMARY KEY,
    learner_key TEXT NOT NULL,
    record_id UUID NOT NULL,
    ref_index INTEGER,
    hyp_index INTEGER,
    error_type TEXT NOT NULL,
    detail JSONB NOT NULL,
    FOREIGN KEY (learner_key, record_id) REFERENCES learning_observations(learner_key, record_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS learning_states (
    learner_key TEXT PRIMARY KEY,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    policy_version TEXT NOT NULL,
    payload JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS learning_feedback (
    feedback_id UUID PRIMARY KEY,
    learner_key TEXT NOT NULL,
    record_id UUID NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('pronunciation','meaning')),
    text TEXT NOT NULL,
    evidence JSONB NOT NULL,
    embedding_status TEXT NOT NULL DEFAULT 'pending' CHECK (embedding_status IN ('pending','ready','failed')),
    FOREIGN KEY (learner_key, record_id) REFERENCES learning_observations(learner_key, record_id) ON DELETE CASCADE,
    UNIQUE (learner_key, feedback_id)
);
CREATE INDEX IF NOT EXISTS learning_feedback_learner ON learning_feedback(learner_key, record_id);
CREATE TABLE IF NOT EXISTS learning_embeddings (
    learner_key TEXT NOT NULL,
    feedback_id UUID NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('feedback','context')),
    model TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    PRIMARY KEY (learner_key, feedback_id, kind),
    FOREIGN KEY (learner_key, feedback_id) REFERENCES learning_feedback(learner_key, feedback_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS learning_design_events (
    id BIGSERIAL PRIMARY KEY,
    learner_key TEXT NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    plan JSONB NOT NULL,
    telemetry JSONB NOT NULL
);
