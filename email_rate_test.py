import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
SMTP_SERVER = 'example.com'
SMTP_PORT = 25
FROM_EMAIL = "email@email.com"
TEST_EMAIL = "your-test-email@email.com"  # Use your own email for testing

def test_single_email(index):
    """Send a single test email and measure response"""
    start_time = time.time()
    
    try:
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = TEST_EMAIL
        msg['Subject'] = f"Rate Limit Test #{index} - {datetime.now()}"
        
        body = f"This is test email #{index} sent at {datetime.now()}"
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.sendmail(FROM_EMAIL, TEST_EMAIL, msg.as_string())
        server.quit()
        
        elapsed = time.time() - start_time
        return True, elapsed, None
        
    except Exception as e:
        elapsed = time.time() - start_time
        return False, elapsed, str(e)

def test_rate_limits():
    """Test email server rate limits with different patterns"""
    print("="*60)
    print("EMAIL SERVER RATE LIMIT TESTING")
    print("="*60)
    
    # Test 1: Burst test - send many emails quickly
    print("\n1. BURST TEST - Sending 20 emails as fast as possible...")
    burst_results = []
    start_time = time.time()
    
    for i in range(20):
        success, elapsed, error = test_single_email(i)
        burst_results.append((success, elapsed, error))
        if success:
            print(f"  Email {i+1}: ✓ ({elapsed:.2f}s)")
        else:
            print(f"  Email {i+1}: ✗ Error: {error}")
        
        # If we start seeing failures, note the pattern
        if not success and "rate" in str(error).lower():
            print(f"  → Rate limit detected at email #{i+1}")
            break
    
    burst_time = time.time() - start_time
    successful = sum(1 for s, _, _ in burst_results if s)
    print(f"\n  Burst test summary: {successful}/20 successful in {burst_time:.2f}s")
    print(f"  Average time per email: {burst_time/len(burst_results):.2f}s")
    
    # Test 2: Concurrent test
    print("\n2. CONCURRENT TEST - Sending 10 emails simultaneously...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(test_single_email, i+100) for i in range(10)]
        concurrent_results = []
        
        for future in as_completed(futures):
            result = future.result()
            concurrent_results.append(result)
    
    successful_concurrent = sum(1 for s, _, _ in concurrent_results if s)
    print(f"  Concurrent test: {successful_concurrent}/10 successful")
    
    # Test 3: Throttled test
    print("\n3. THROTTLED TEST - Sending 10 emails with 2-second delays...")
    throttled_results = []
    
    for i in range(10):
        success, elapsed, error = test_single_email(i+200)
        throttled_results.append((success, elapsed, error))
        if success:
            print(f"  Email {i+1}: ✓ ({elapsed:.2f}s)")
        else:
            print(f"  Email {i+1}: ✗ Error: {error}")
        time.sleep(2)  # 2-second delay between emails
    
    successful_throttled = sum(1 for s, _, _ in throttled_results if s)
    print(f"\n  Throttled test: {successful_throttled}/10 successful")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS & RECOMMENDATIONS:")
    print("="*60)
    
    # Determine rate limits based on results
    if successful < 20:
        print(f"⚠️  Rate limiting detected: Only {successful}/20 burst emails succeeded")
        print(f"   Recommended: Add delays between emails or limit concurrent sends")
    else:
        print("✓  No obvious rate limiting detected for burst sends")
    
    if successful_concurrent < 10:
        print(f"⚠️  Concurrent connection limit: Only {successful_concurrent}/10 concurrent emails succeeded")
        print(f"   Recommended: Limit concurrent workers to {successful_concurrent} or less")
    else:
        print("✓  Server handles 10 concurrent connections well")
    
    print("\nRecommended settings for your main script:")
    if successful < 20 or successful_concurrent < 10:
        print("  - max_workers: 4-6 (conservative)")
        print("  - Add 0.5-1 second delay between emails")
    else:
        print("  - max_workers: 8-12 (aggressive)")
        print("  - No delay needed between emails")

def test_server_connection_info():
    """Get detailed information about the email server"""
    print("\n" + "="*60)
    print("EMAIL SERVER INFORMATION")
    print("="*60)
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.set_debuglevel(1)  # Enable debug output
        
        # Get server features
        server.ehlo()
        features = server.esmtp_features
        
        print("\nServer Features:")
        for feature, params in features.items():
            print(f"  - {feature}: {params}")
        
        # Check for specific limits
        if 'size' in features:
            print(f"\n  Max message size: {features['size']} bytes")
        
        server.quit()
        
    except Exception as e:
        print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    # First, test server information
    test_server_connection_info()
    
    # Then test rate limits
    print("\n⚠️  WARNING: This will send ~40 test emails to:", TEST_EMAIL)
    response = input("Continue? (y/n): ")
    
    if response.lower() == 'y':
        test_rate_limits()
    else:
        print("Test cancelled.")
