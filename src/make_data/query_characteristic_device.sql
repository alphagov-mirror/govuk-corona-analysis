SELECT
  EXTRACT(DATE FROM visitStartTimestamp)
  ,COUNT(DISTINCT CONCAT(session_id, "-", hit_Number)) AS pageviews
FROM `govuk-xgov.xgov_data_access.Basetable_corona`
WHERE type = "PAGE"
  AND pagePath LIKE "%coronavirus-covid-19-uk-government-response%"
GROUP BY EXTRACT(DATE FROM visitStartTimestamp)
