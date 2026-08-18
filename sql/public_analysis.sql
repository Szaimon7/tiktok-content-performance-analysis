-- Analysis-ready view using anonymized identifiers and content scores
-- The view excludes private documentation and original identifiers

CREATE OR REPLACE
SQL SECURITY INVOKER
VIEW public_analysis AS
SELECT
    v.public_video_id AS video_id,
    v.duration_seconds,
    v.views,
    v.likes,
    v.comments,
    v.shares,
    vm.like_rate,
    vm.comment_rate,
    vm.share_rate,
    vm.engagement_rate,
    vm.active_engagement_rate,
    s.public_speaker_id AS speaker,
    f.format_public AS format,
    f.format_family,
    f.location_type,
    vs.hook_score,
    vs.context_independence_level,
    vs.spotness_score,
    vs.naturalness_score,
    vs.emotion_score,
    vs.concreteness_score,
    vs.has_punchline,
    vs.punchline_strength,
    vs.backlash_risk_score
FROM videos AS v
JOIN video_metrics AS vm
    ON v.video_id = vm.video_id
LEFT JOIN speakers AS s
    ON v.speaker_id = s.speaker_id
LEFT JOIN formats AS f
    ON v.format_id = f.format_id
JOIN video_scores AS vs
    ON v.video_id = vs.video_id;