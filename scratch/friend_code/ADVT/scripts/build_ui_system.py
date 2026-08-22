# Build script for SYNTHGUARD UI
import os
import sys
import re

print("Initializing SYNTHGUARD UI Component Sanitizer & Builder...")

def sanitize_directory(dir_path):
    count = 0
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f.endswith('.py'):
                p = os.path.join(root, f)
                with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                    content = fp.read()
                
                # Strip all leading or isolated question mark sequences
                cleaned = content
                # Multiple question marks e.g. ???, ??
                cleaned = re.sub(r'\?{2,}\s*', '', cleaned)
                # Specific mangled tokens
                cleaned = cleaned.replace('(?)', '(Epsilon)')
                cleaned = cleaned.replace('(? Sweep)', '(Epsilon Sweep)')
                cleaned = cleaned.replace('(?, ?)-DP', '(Epsilon, Delta)-DP')
                cleaned = cleaned.replace('(?, ?)', '(Epsilon, Delta)')
                cleaned = cleaned.replace('R?nyi', 'Renyi')
                cleaned = cleaned.replace('? Ingested', 'Ingested')
                cleaned = cleaned.replace('? Ready', 'Ready')
                cleaned = cleaned.replace('? Mathematically', 'Mathematically')
                cleaned = cleaned.replace('? Domain', 'Domain')
                cleaned = cleaned.replace('? 3.', '3.')
                cleaned = cleaned.replace('? (', '(')
                cleaned = cleaned.replace('? DROPPED', 'DROPPED')
                cleaned = cleaned.replace('? INVERTIBLE', 'INVERTIBLE')
                cleaned = cleaned.replace('? ONE-HOT', 'ONE-HOT')
                cleaned = cleaned.replace('? HIGH-CARDINALITY', 'HIGH-CARDINALITY')
                cleaned = cleaned.replace('? AUTO-STRIPPED', 'AUTO-STRIPPED')
                cleaned = cleaned.replace('? Safe', 'Safe')
                cleaned = cleaned.replace('? CONVERGED', 'CONVERGED')
                cleaned = cleaned.replace('? PASSED', 'PASSED')
                cleaned = cleaned.replace('? PASS', 'PASS')
                cleaned = cleaned.replace('? Zero NaN', 'Zero NaN')
                cleaned = cleaned.replace('? Leak', 'Leak')
                cleaned = cleaned.replace('? Automated', 'Automated')
                cleaned = cleaned.replace('? No Memorization', 'No Memorization')
                cleaned = cleaned.replace('? High Utility', 'High Utility')
                cleaned = cleaned.replace('? HIPAA', 'HIPAA')
                cleaned = cleaned.replace('? Screen', 'Screen')
                cleaned = cleaned.replace('? **Bivariate', '**Bivariate')
                cleaned = cleaned.replace('? {status}', '{status}')
                cleaned = cleaned.replace('Target ?:', 'Target Delta:')
                cleaned = cleaned.replace('Target ?', 'Target Delta')
                cleaned = cleaned.replace('Cryptographic ?', 'Cryptographic Delta')
                cleaned = cleaned.replace('Cumulative ? Spent', 'Cumulative Epsilon Spent')
                cleaned = cleaned.replace('Final Spent ?', 'Final Spent Epsilon')
                cleaned = cleaned.replace('Privacy Guarantee:    ?', 'Privacy Guarantee:    Epsilon')
                cleaned = cleaned.replace('(? = 1.0 ? 10', '(Delta = 1.0e-4')
                cleaned = cleaned.replace('Noise Multiplier:     ?', 'Noise Multiplier:     Sigma')
                cleaned = cleaned.replace('(?=1.0)', '(Epsilon=1.0)')
                cleaned = cleaned.replace('(?={active_eps})', '(Epsilon={active_eps})')
                cleaned = cleaned.replace('(? = 0.1)', '(Epsilon = 0.1)')
                cleaned = cleaned.replace('(? = 1.0)', '(Epsilon = 1.0)')
                cleaned = cleaned.replace('(? = 10.0)', '(Epsilon = 10.0)')
                cleaned = cleaned.replace('45 CFR ?', '45 CFR §')
                cleaned = cleaned.replace('45 CFR ?164', '45 CFR §164')
                cleaned = cleaned.replace('AUC ? 0.50', 'AUC <= 0.50')
                cleaned = cleaned.replace('time_in_hospital ? [1, 14]', 'time_in_hospital in [1, 14]')
                cleaned = cleaned.replace('Lower ? provides', 'Lower epsilon provides')
                cleaned = cleaned.replace('recommendation: ? ? [0.5, 1.5]', 'recommendation: epsilon in [0.5, 1.5]')
                cleaned = cleaned.replace('? ? {target_eps', 'Epsilon approx {target_eps')
                cleaned = cleaned.replace('1.0 ? 10', '1.0 x 10^-4 ')
                cleaned = cleaned.replace('Target Delta (Epsilon):', 'Target Delta:')
                cleaned = cleaned.replace('Training ?', 'Training ->')
                cleaned = cleaned.replace('Sanitization ?', 'Sanitization ->')
                cleaned = cleaned.replace('Dashboard ?', 'Dashboard ->')
                cleaned = cleaned.replace('("3. Synthesis & Guardrails", "?")', '("3. Synthesis & Guardrails", "")')
                cleaned = cleaned.replace('page_icon="???"', 'page_icon="SG"')
                cleaned = cleaned.replace('page_icon="?"', 'page_icon="SG"')
                cleaned = cleaned.replace('>???<', '><')
                cleaned = cleaned.replace('>??<', '><')
                cleaned = cleaned.replace('>?<', '><')
                cleaned = cleaned.replace('>???', '>')
                cleaned = cleaned.replace('>??', '>')
                cleaned = cleaned.replace('> ?', '>')
                cleaned = cleaned.replace('???', '')
                cleaned = cleaned.replace('??', '')
                
                if cleaned != content:
                    with open(p, 'w', encoding='utf-8') as fp:
                        fp.write(cleaned)
                    count += 1
                    print(f"Sanitized: {p}")
    print(f"Total files sanitized: {count}")

if __name__ == "__main__":
    sanitize_directory("ui")

