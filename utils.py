import datetime

def is_active_policy(policy_dict,date):
    """
    Example logic: 
      - We consider a policy active if 'resContractStatus' == '정상'
      - Optionally also check date range (commEndDate in the future).
        But your data uses strings like '20200214'. You can parse them as needed.
    """
    #if policy_dict.get('resContractStatus') != '정상':
    #    return False
    
    # Example: parse commEndDate and check if still in the future
    # (You can skip this if you just want to filter by '정상')
    end_date_str = policy_dict.get('commEndDate', '')  # e.g. "20200214"
    if not end_date_str:
        end_date_str = policy_dict.get('resCoverageLists','')[0].get('commEndDate', '')
        
    start_date_str = policy_dict.get('commStartDate', '')  # e.g. "20200214"
    if not start_date_str:
        start_date_str = policy_dict.get('resCoverageLists','')[0].get('commStartDate', '')
        
    
    # Try to parse year-month-day
    try:
        end_date = datetime.datetime.strptime(end_date_str, "%Y%m%d").date()
        start_date = datetime.datetime.strptime(start_date_str, "%Y%m%d").date()
        date = datetime.datetime.strptime(date,"%Y%m%d").date()
        #today = datetime.date.today()  # or any reference date you want
        return (end_date >= date) and (date >= start_date)
    except ValueError:
        # If date format is wrong or missing, skip
        return False
def is_coverage_active(coverage_dict: dict,date) -> bool:
    """
    Decide if a coverage line is 'active':
      - resCoverageStatus == '정상'
      - commEndDate >= today (if commEndDate exists)
    """
    #if coverage_dict.get('resCoverageStatus') != '정상':
    #    return False

    end_date_str = coverage_dict.get('commEndDate', '')
    start_date_str = coverage_dict.get('commStartDate', '')

    try:
        end_date = datetime.datetime.strptime(end_date_str, "%Y%m%d").date()
        start_date = datetime.datetime.strptime(start_date_str, "%Y%m%d").date()
        date = datetime.datetime.strptime(date, "%Y%m%d").date()
        return (end_date >= date) and (date >= start_date)
    except ValueError:
        # If date parsing fails, consider coverage inactive or handle differently
        return False    


def gather_active_contracts(demo_data: dict,date:str) -> list:
    """
    1) Gathers contracts from both flat-rate and actual-loss lists.
    2) Filters out non-'정상' contracts.
    3) Adds a key 'contractType' with either 'flatRate' or 'actualLoss'
       so we can decide which renderer to use later.
    4) Returns a single list.
    """
    data_section = demo_data.get('data', {})

    # Flat-Rate
    flat_rate_list = data_section.get('resFlatRateContractList', [])
    active_flat = []
    for c in flat_rate_list:
        if is_active_policy(c,date):
            contract = dict(c)  # Make a shallow copy
            contract['contractType'] = 'flatRate'
            active_flat.append(contract)

    # Actual-Loss
    actual_loss_list = data_section.get('resActualLossContractList', [])
    active_actual = []
    for c in actual_loss_list:
        if is_active_policy(c,date):
            contract = dict(c)
            contract['contractType'] = 'actualLoss'
            active_actual.append(contract)

    return active_flat + active_actual

def render_policy_as_table_flat(policy_dict,date):
    """
    Returns a multiline string for a single policy in your desired format.
    """
    # Basic policy fields
    company_name = policy_dict.get('resCompanyNm', 'Unknown')
    insurance_name = policy_dict.get('resInsuranceName', 'Unknown')
    contract_status = policy_dict.get('resContractStatus', 'Unknown')
    policy_number = policy_dict.get('resPolicyNumber', 'Unknown')
    policyholder = policy_dict.get('resContractor', 'Unknown')
    start_date = policy_dict.get('commStartDate', 'Unknown')
    end_date   = policy_dict.get('commEndDate', 'Unknown')
    payment_cycle = policy_dict.get('resPaymentCycle', 'Unknown')
    payment_period = policy_dict.get('resPaymentPeriod', 'Unknown')
    premium = policy_dict.get('resPremium', 'Unknown')

    # Convert date format YYYYMMDD -> YYYY.MM.DD for nicer display
    def pretty_date(yyyymmdd):
        if len(yyyymmdd) == 8:
            return f"{yyyymmdd[0:4]}.{yyyymmdd[4:6]}.{yyyymmdd[6:8]}"
        return yyyymmdd

    start_date_str = pretty_date(start_date)
    end_date_str   = pretty_date(end_date)

    # Gather coverage rows
    coverage_list = policy_dict.get('resCoverageLists', [])

    # Build table rows
    coverage_rows = []
    for cov in coverage_list:
        coverage_type  = cov.get('resAgreementType', '')
        coverage_name  = cov.get('resCoverageName', '')
        coverage_stat  = cov.get('resCoverageStatus', '')
        coverage_amt   = cov.get('resCoverageAmount', '0')
        # Format coverage amount with commas
        try:
            coverage_amt = f"{int(coverage_amt):,}"
        except:
            pass

        # Example row: coverage_type, coverage_name, coverage_stat, coverage_amt
        coverage_rows.append(
            f"| {coverage_type:<30} "
            f"| {coverage_name:<60} "
            f"| {coverage_stat:<6} "
            f"| {coverage_amt:>10} |"
        )

    # Construct final output
    result_lines = []

    result_lines.append("[정액보험]")
    result_lines.append(f"보험사 : {company_name}")
    result_lines.append(f"보험명 : {insurance_name}")
    result_lines.append(f"보장상태 : {contract_status}")
    result_lines.append(f"증권번호 : {policy_number}")
    result_lines.append(f"계약자 : {policyholder}")
    result_lines.append(f"보장시작일 : {start_date_str}")
    result_lines.append(f"보장종료일 : {end_date_str}")
    result_lines.append(f"납입 주기 : {payment_cycle}")
    result_lines.append(f"납입 기간 : {payment_period} 년")
    result_lines.append(f"1회 보험료 : {premium} 원")
    result_lines.append("보험 내용:")
    result_lines.append("| 보장구분                 | 보장명                                               | 보장상태 | 보장금액 (원) |")
    result_lines.append("|-------------------------------|------------------------------------------------------------|--------|----------------|")

    # Append coverage rows
    result_lines.extend(coverage_rows)

    # Join them all with newlines
    return "\n".join(result_lines) + "\n"

def render_policy_as_table_actual(contract_dict: dict,date) -> str:
    """
    Renders an Actual-Loss contract into a multiline string.
    We skip coverage lines that are not active (resCoverageStatus != '정상' or end date in past).
    We'll *show* each coverage's commStartDate/commEndDate in additional columns.
    """
    company_name   = contract_dict.get('resCompanyNm', 'Unknown')
    insurance_name = contract_dict.get('resInsuranceName', 'Unknown')
    contract_status = contract_dict.get('resContractStatus', 'Unknown')
    policy_number  = contract_dict.get('resPolicyNumber', 'Unknown')
    policyholder   = contract_dict.get('resContractor', 'Unknown')

    # We can keep these if present, else 'N/A'
    payment_cycle  = contract_dict.get('resPaymentCycle', 'N/A')
    payment_period = contract_dict.get('resPaymentPeriod', 'N/A')
    premium        = contract_dict.get('resPremium', 'N/A')

    # For actual-loss, let's not rely on top-level `commStartDate/commEndDate`.
    # We'll show coverage-level dates in the table.
    lines = []
    lines.append("[실손보험]")
    lines.append(f"보험사: {company_name}")
    lines.append(f"보험명: {insurance_name}")
    lines.append(f"보장상태: {contract_status}")
    lines.append(f"증권번호: {policy_number}")
    lines.append(f"계약자: {policyholder}")
    lines.append(f"납입 주기: {payment_cycle}")
    lines.append(f"납입 기간: {payment_period} years")
    lines.append(f"1회 보험료: {premium} KRW")

    coverage_list  = contract_dict.get('resCoverageLists', [])

    lines.append("보험 내용 (실손):")
    # We'll add two extra columns for Start / End date
    lines.append("| 보장구분                 | 보장명                               | 보장시작일  | 보장종료일    | 보장상태 | 보장금액 (원) |")
    lines.append("|-------------------------------|--------------------------------------------|------------|------------|--------|----------------|")
    for cov in coverage_list:
        if not is_coverage_active(cov,date):
            # Skip coverage that is not active (status != 정�상 or end date < today)
            continue

        coverage_type = cov.get('resType', 'N/A')
        coverage_name = cov.get('resCoverageName', 'N/A')
        coverage_stat = cov.get('resCoverageStatus', 'N/A')
        coverage_amt  = cov.get('resCoverageAmount', '0')

        start_date = cov.get('commStartDate', 'N/A')
        end_date   = cov.get('commEndDate', 'N/A')
        
        def pretty_date(yyyymmdd):
            if len(yyyymmdd) == 8:
                return f"{yyyymmdd[0:4]}.{yyyymmdd[4:6]}.{yyyymmdd[6:8]}"
            return yyyymmdd

        start_date_str = pretty_date(start_date)
        end_date_str   = pretty_date(end_date)

        # Format coverage_amt with commas
        try:
            coverage_amt = f"{int(coverage_amt):,}"
        except:
            pass

        row = (
            f"| {coverage_type:<30}"
            f" | {coverage_name:<40}"
            f" | {start_date_str:<10}"
            f" | {end_date_str:<10}"
            f" | {coverage_stat:<6}"
            f" | {coverage_amt:>14} |"
        )
        lines.append(row)

    return "\n".join(lines) + "\n"

def process_and_print_active_policies(demo_data,date) -> str:
    """
    Filters for active policies, then builds and returns a
    single multiline string containing all those policies.
    """
    active_policies = gather_active_contracts(demo_data,date)
    
    if not active_policies:
        return "No active policies found."
    
    
    results = []
    for i, policy in enumerate(active_policies, start=1):
        if policy['contractType'] == 'flatRate':
            table_str = render_policy_as_table_flat(policy,date)
        # Add a section header + the table + a separator line
            block = f"[Insurance #{i}]\n{table_str}\n" + ("-" * 10)
            results.append(block)
        elif policy['contractType'] == 'actualLoss':
            table_str = render_policy_as_table_actual(policy,date)
        # Add a section header + the table + a separator line
            block = f"[Insurance #{i}]\n{table_str}\n" + ("-" * 10)
            results.append(block)
            
    
    # Combine everything into one big string
    final_output = "\n\n".join(results)
    #print(final_output)
    return final_output