import datetime

def is_active_policy(policy_dict):
    """
    Example logic: 
      - We consider a policy active if 'resContractStatus' == '정상'
      - Optionally also check date range (commEndDate in the future).
        But your data uses strings like '20200214'. You can parse them as needed.
    """
    if policy_dict.get('resContractStatus') != '정상':
        return False
    
    # Example: parse commEndDate and check if still in the future
    # (You can skip this if you just want to filter by '정상')
    end_date_str = policy_dict.get('commEndDate', '')  # e.g. "20200214"
    if not end_date_str:
        return False
    
    # Try to parse year-month-day
    try:
        end_date = datetime.datetime.strptime(end_date_str, "%Y%m%d").date()
        today = datetime.date.today()  # or any reference date you want
        return end_date >= today
    except ValueError:
        # If date format is wrong or missing, skip
        return False

def extract_active_contracts(data: dict):
    """
    Return a list of active (정상) flat-rate contracts.
    """
    contracts_flat = data.get('data', {}).get('resFlatRateContractList', [])
    contracts_loss = data.get('data', {}).get('resActualLossContractList', [])
    
    contracts = contracts_flat + contracts_loss
    active = []
    for c in contracts:
        if is_active_policy(c):
            active.append(c)
    return active

def render_policy_as_table(policy_dict):
    """
    Returns a multiline string for a single policy in your desired format.
    """
    # Basic policy fields
    company_name = policy_dict.get('resCompanyNm', 'Unknown')
    insurance_name = policy_dict.get('resInsuranceName', 'Unknown')
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

    result_lines.append(f"보험사 : {company_name}")
    result_lines.append(f"보험명 : {insurance_name}")
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

def process_and_print_active_policies(demo_data) -> str:
    """
    Filters for active policies, then builds and returns a
    single multiline string containing all those policies.
    """
    active_policies = extract_active_contracts(demo_data)
    
    if not active_policies:
        return "No active policies found."
    
    results = []
    for i, policy in enumerate(active_policies, start=1):
        table_str = render_policy_as_table(policy)
        # Add a section header + the table + a separator line
        block = f"[Insurance #{i}]\n{table_str}\n" + ("-" * 10)
        results.append(block)
    
    # Combine everything into one big string
    final_output = "\n\n".join(results)
    #print(final_output)
    return final_output
