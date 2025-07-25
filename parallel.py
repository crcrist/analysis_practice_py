import tableauserverclient as tsc
from dotenv import load_dotenv
import json
import os 
import PyPDF2
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from typing import List, Dict, Tuple

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from google.cloud import bigquery

# Load environment variables
load_dotenv()

# Configuration - Modified to support multiple PAT tokens
tableau_tokens = [
    {"name": "token1", "value": "faketokenvalue1"},
    {"name": "token2", "value": "faketokenvalue2"},
    {"name": "token3", "value": "faketokenvalue3"},
    {"name": "token4", "value": "faketokenvalue4"},
    # Add more tokens as needed
]
outlook_email = "email@email.com"
outlook_pw = "password"
workbook_name = "workboook"
workbook_view_name = ["view_name1", "view_name2", "view_name3"]
ssl_certificate = "fakecert"

# Thread-safe Tableau connection pool with multiple PAT tokens
class TableauConnectionPool:
    def __init__(self, tokens_list):
        self.connections = []
        self.lock = threading.Lock()
        
        # Create one authenticated server connection per PAT token
        for token in tokens_list:
            auth = tsc.PersonalAccessTokenAuth(token["name"], token["value"])
            server = tsc.Server('faketableauserver.com', use_server_version=True)
            server.add_http_options({'verify': ssl_certificate})
            with server.auth.sign_in(auth):
                self.connections.append(server)
        
        print(f"Initialized connection pool with {len(self.connections)} PAT tokens")
    
    def get_connection(self):
        with self.lock:
            if self.connections:
                return self.connections.pop()
        # Wait and retry if no connections available
        time.sleep(0.5)
        return self.get_connection()
    
    def return_connection(self, conn):
        with self.lock:
            self.connections.append(conn)

# Initialize connection pool with multiple PAT tokens
connection_pool = TableauConnectionPool(tableau_tokens)

# BigQuery setup
key_path = 'fake/path'
os.environ['cred'] = key_path
client = bigquery.Client(project='prod')

today = datetime.now() - timedelta(days=1)
subject_date = today.strftime('%b %d')

def get_all_users():
    """Fetch all users without the 50 user limit"""
    query = """
        SELECT DISTINCT 
            user, email, filter
        FROM table
        -- Remove the WHERE clause to get all users
        -- Or adjust as needed: WHERE user_id > 0
    """
    
    query_job = client.query(query)
    results = list(query_job.result())
    
    # Convert to list of dictionaries for easier handling
    users = []
    for r in results:
        users.append({
            'user': r[0],
            'email': r[1],
            'filter': r[2]
        })
    
    print(f"Total users to process: {len(users)}")
    return users

def send_email(pdf_file, from_addr, to_addr, user):
    """Send email with PDF attachment"""
    msg = MIMEMultipart()
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = f"example subject {user}"
    
    body = f"""example body {user}"""
    msg.attach(MIMEText(body, 'plain'))
    
    with open(pdf_file, "rb") as f:
        attach = MIMEApplication(f.read(), _subtype="pdf")
    
    attach.add_header('Content-Disposition', f"attachment; filename={os.path.basename(pdf_file)}")
    msg.attach(attach)
    
    server = smtplib.SMTP('example.com', 25)
    server.starttls()
    
    text = msg.as_string()
    server.sendmail(from_addr, to_addr, text)
    server.quit()

def process_user(user_data: Dict, workbook_id: str) -> Tuple[str, bool, str]:
    """Process a single user's PDF generation and email sending"""
    user = user_data['user']
    email = user_data['email']
    filter_value = user_data['filter']
    
    if not email:
        return user, False, "No email address"
    
    # Get a connection from the pool
    server = connection_pool.get_connection()
    
    try:
        # Get the workbook
        workbook = server.workbooks.get_by_id(workbook_id)
        server.workbooks.populate_views(workbook)
        
        pdf_files = []
        
        for view in workbook.views:
            if view.name in workbook_view_name:
                # Create PDF request options
                pdf_req_option = tsc.PDFRequestOptions(
                    page_type=tsc.PDFRequestOptions.PageType.Unspecified,
                    orientation=tsc.PDFRequestOptions.Orientation.Portrait
                )
                pdf_req_option.vf('Parameters.Filter', filter_value)
                
                # Generate PDF
                server.views.populate_pdf(view, pdf_req_option)
                
                if not view.pdf:
                    continue
                
                output_dir = '/tmp'
                pdf_filename = os.path.join(output_dir, f'{view.name.replace(" ", "_")}_{user}_recap.pdf')
                
                with open(pdf_filename, 'wb') as f:
                    f.write(view.pdf)
                
                pdf_files.append(pdf_filename)
        
        if not pdf_files:
            return user, False, "No PDFs generated"
        
        # Merge PDFs
        merger = PyPDF2.PdfMerger()
        for pdf_file in pdf_files:
            merger.append(pdf_file)
        
        combined_pdf = os.path.join('/tmp', f'report_{user}.pdf')
        merger.write(combined_pdf)
        merger.close()
        
        # Send email
        send_email(combined_pdf, outlook_email, email, user)
        
        # Cleanup
        for pdf_file in pdf_files:
            os.remove(pdf_file)
        os.remove(combined_pdf)
        
        return user, True, "Success"
        
    except Exception as e:
        return user, False, f"Error: {str(e)}"
    
    finally:
        # Return connection to pool
        connection_pool.return_connection(server)

def main():
    """Main execution function"""
    start_time = time.time()
    
    # Get all users
    users = get_all_users()
    
    # Get workbook ID once (more efficient than searching by name repeatedly)
    server = connection_pool.get_connection()
    workbook_id = None
    
    try:
        for workbook in tsc.Pager(server.workbooks):
            if workbook.name == workbook_name:
                workbook_id = workbook.id
                break
    finally:
        connection_pool.return_connection(server)
    
    if not workbook_id:
        print(f"Workbook '{workbook_name}' not found!")
        return
    
    # Process users concurrently
    successful = 0
    failed = 0
    
    # Use ThreadPoolExecutor for concurrent processing
    # Set max_workers to the number of PAT tokens available
    max_workers = min(len(tableau_tokens), 8)  # Cap at 8 or number of tokens
    print(f"Using {max_workers} concurrent workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_user = {
            executor.submit(process_user, user, workbook_id): user 
            for user in users
        }
        
        # Process completed tasks
        for future in as_completed(future_to_user):
            user_data = future_to_user[future]
            try:
                user_name, success, message = future.result()
                if success:
                    successful += 1
                    print(f"✓ Sent email to {user_name} at {user_data['email']}")
                else:
                    failed += 1
                    print(f"✗ Failed for {user_name}: {message}")
            except Exception as exc:
                failed += 1
                print(f"✗ Exception for {user_data['user']}: {exc}")
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*50)
    print(f"Processing complete!")
    print(f"Total users: {len(users)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Average time per user: {duration/len(users):.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    main()
