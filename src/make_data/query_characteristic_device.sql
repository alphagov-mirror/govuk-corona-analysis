SELECT 
  visitStartTimestamp
  ,deviceCategory
  ,COUNT(DISTINCT CONCAT(session_id, "-", hit_Number)) AS pageviews
FROM `govuk-xgov.InsightsDataset.covid19ukgovresponse`
WHERE type = "PAGE"
GROUP BY visitStartTimestamp
  ,deviceCategory
