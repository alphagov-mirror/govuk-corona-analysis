SELECT 
  visitStartTimestamp
  ,deviceCategory
  ,COUNT(DISTINCT CONCAT(CAST(fullVisitorId AS STRING), "-", CAST(visitId AS STRING), "-", CAST(hit_Number AS STRING))) AS pageviews 
FROM `govuk-xgov.InsightsDataset.covid19ukgovresponse`
WHERE type = "PAGE"
GROUP BY visitStartTimestamp
  ,deviceCategory
