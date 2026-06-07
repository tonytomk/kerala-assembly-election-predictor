#!/usr/bin/env python3
import json
from pathlib import Path

json_path = Path(__file__).parent.parent / "dashboard" / "src" / "data" / "historical_vote_shares.json"

with open(json_path, 'r') as f:
    data = json.load(f)

issues = []

print("Validating vote share totals for each constituency/year...\n")
print(f"{'Constituency':<25} {'Year':<6} {'UDF':<8} {'LDF':<8} {'NDA':<8} {'OTHER':<8} {'Total':<8} {'Status':<15}")
print("=" * 100)

for constituency in data:
    c_num = constituency['constituency_number']
    c_name = constituency['constituency_name']
    
    # Group by year
    by_year = {}
    for row in constituency['rows']:
        year = row['year']
        party = row['party']
        share = row['vote_share']
        
        if year not in by_year:
            by_year[year] = {}
        by_year[year][party] = share
    
    # Check totals for each year
    for year in sorted(by_year.keys()):
        parties = by_year[year]
        udf = parties.get('UDF', 0)
        ldf = parties.get('LDF', 0)
        nda = parties.get('NDA', 0)
        other = parties.get('OTHER', 0)
        
        total = udf + ldf + nda + other
        
        status = "OK" if 99.5 <= total <= 100.5 else "ERROR"
        
        if not (99.5 <= total <= 100.5):
            issues.append({
                'constituency': c_name,
                'number': c_num,
                'year': year,
                'total': total,
                'udf': udf,
                'ldf': ldf,
                'nda': nda,
                'other': other
            })
        
        print(f"{c_name:<25} {year:<6} {udf:>7.2f} {ldf:>7.2f} {nda:>7.2f} {other:>7.2f} {total:>7.2f} {status:<15}")

print("\n" + "=" * 100)
print(f"\nTotal constituencies checked: {len(data)}")
print(f"Issues found: {len(issues)}")

if issues:
    print("\nProblematic entries:")
    for issue in issues:
        print(f"\n  {issue['constituency']} (#{issue['number']}) - {issue['year']}")
        print(f"    UDF: {issue['udf']:.2f}%")
        print(f"    LDF: {issue['ldf']:.2f}%")
        print(f"    NDA: {issue['nda']:.2f}%")
        print(f"    OTHER: {issue['other']:.2f}%")
        print(f"    TOTAL: {issue['total']:.2f}% (Expected ~100%)")
else:
    print("\nAll vote shares are properly balanced!")
