#!/usr/bin/env python3
"""
Scheduled Reporting System for Stage 6.5
Automated report generation with configurable schedules, email delivery, and notification systems.
"""

import logging
import json
import smtplib
import schedule
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

# Email libraries
try:
    from email.mime.application import MIMEApplication
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    logging.warning("Email support not fully available")

# Import our modules
from report_export import get_report_export_manager, ReportRequest
from config_manager import get_config_manager

class ScheduleFrequency(Enum):
    """Report schedule frequencies"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"

class ReportStatus(Enum):
    """Report generation status"""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    SENT = "sent"

class DeliveryMethod(Enum):
    """Report delivery methods"""
    EMAIL = "email"
    FILE_SYSTEM = "file_system"
    WEBHOOK = "webhook"
    FTP = "ftp"

@dataclass
class EmailConfiguration:
    """Email server configuration"""
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    use_tls: bool = True
    use_ssl: bool = False
    from_address: str = ""
    reply_to: str = ""

@dataclass
class DeliveryConfiguration:
    """Report delivery configuration"""
    method: DeliveryMethod
    email_config: Optional[EmailConfiguration] = None
    file_path: Optional[str] = None
    webhook_url: Optional[str] = None
    ftp_config: Optional[Dict[str, Any]] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScheduledReportConfig:
    """Scheduled report configuration"""
    report_id: str
    report_name: str
    template_id: str
    frequency: ScheduleFrequency
    schedule_time: str  # e.g., "08:00", "Monday 09:00", "*/6"
    enabled: bool = True
    timerange_days: int = 1  # How many days of data to include
    output_format: str = "pdf"  # pdf, excel, both
    delivery_config: DeliveryConfiguration = None
    recipients: List[str] = field(default_factory=list)
    custom_params: Dict[str, Any] = field(default_factory=dict)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    failure_count: int = 0

@dataclass
class ReportExecution:
    """Report execution record"""
    execution_id: str
    report_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ReportStatus = ReportStatus.PENDING
    output_files: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    delivery_status: Dict[str, bool] = field(default_factory=dict)
    execution_time_seconds: float = 0.0

class EmailManager:
    """Manages email sending functionality"""
    
    def __init__(self, email_config: EmailConfiguration):
        self.config = email_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not EMAIL_AVAILABLE:
            raise ImportError("Email support not available")
    
    def send_report_email(self, recipients: List[str], subject: str, body: str, 
                         attachments: List[str] = None) -> bool:
        """Send report via email with attachments"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.from_address or self.config.username
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject
            
            if self.config.reply_to:
                msg['Reply-To'] = self.config.reply_to
            
            # Add body
            msg.attach(MIMEText(body, 'html'))
            
            # Add attachments
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'rb') as attachment:
                                part = MIMEBase('application', 'octet-stream')
                                part.set_payload(attachment.read())
                                encoders.encode_base64(part)
                                part.add_header(
                                    'Content-Disposition',
                                    f'attachment; filename= {os.path.basename(file_path)}'
                                )
                                msg.attach(part)
                        except Exception as e:
                            self.logger.warning(f"Could not attach file {file_path}: {e}")
                    else:
                        self.logger.warning(f"Attachment file not found: {file_path}")
            
            # Send email
            if self.config.use_ssl:
                server = smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port)
            else:
                server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
                if self.config.use_tls:
                    server.starttls()
            
            server.login(self.config.username, self.config.password)
            text = msg.as_string()
            server.sendmail(self.config.username, recipients, text)
            server.quit()
            
            self.logger.info(f"Email sent successfully to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test email server connection"""
        try:
            if self.config.use_ssl:
                server = smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port)
            else:
                server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
                if self.config.use_tls:
                    server.starttls()
            
            server.login(self.config.username, self.config.password)
            server.quit()
            
            self.logger.info("Email server connection test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Email server connection test failed: {e}")
            return False

class ReportScheduler:
    """Manages scheduled report execution"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scheduled_reports: Dict[str, ScheduledReportConfig] = {}
        self.report_executions: Dict[str, ReportExecution] = {}
        self.execution_history: List[ReportExecution] = []
        
        # Threading
        self.scheduler_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Managers
        self.report_manager = get_report_export_manager()
        self.config_manager = get_config_manager()
        
        # Email managers by config
        self.email_managers: Dict[str, EmailManager] = {}
        
        # Load configurations
        self._load_configurations()
        
        self.logger.info("Report scheduler initialized")
    
    def _load_configurations(self):
        """Load scheduled report configurations"""
        try:
            config_path = Path("telemetry_monitor/scheduled_reports.json")
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    data = json.load(f)
                
                # Load scheduled reports
                for report_data in data.get('scheduled_reports', []):
                    # Convert enums
                    report_data['frequency'] = ScheduleFrequency(report_data['frequency'])
                    
                    if 'delivery_config' in report_data and report_data['delivery_config']:
                        delivery_data = report_data['delivery_config']
                        delivery_data['method'] = DeliveryMethod(delivery_data['method'])
                        
                        if 'email_config' in delivery_data and delivery_data['email_config']:
                            email_data = delivery_data['email_config']
                            delivery_data['email_config'] = EmailConfiguration(**email_data)
                        
                        report_data['delivery_config'] = DeliveryConfiguration(**delivery_data)
                    
                    # Parse datetime fields
                    if report_data.get('last_run'):
                        report_data['last_run'] = datetime.fromisoformat(report_data['last_run'])
                    if report_data.get('next_run'):
                        report_data['next_run'] = datetime.fromisoformat(report_data['next_run'])
                    
                    config = ScheduledReportConfig(**report_data)
                    self.scheduled_reports[config.report_id] = config
                
                # Load email configurations
                for email_data in data.get('email_configurations', []):
                    config_id = email_data.pop('config_id')
                    email_config = EmailConfiguration(**email_data)
                    self.email_managers[config_id] = EmailManager(email_config)
                
                self.logger.info(f"Loaded {len(self.scheduled_reports)} scheduled report configurations")
            
            else:
                # Create default configurations
                self._create_default_configurations()
                
        except Exception as e:
            self.logger.error(f"Error loading scheduled report configurations: {e}")
            self._create_default_configurations()
    
    def _create_default_configurations(self):
        """Create default scheduled report configurations"""
        try:
            # Daily system health report
            daily_report = ScheduledReportConfig(
                report_id="daily_health",
                report_name="Daily System Health Report",
                template_id="system_health",
                frequency=ScheduleFrequency.DAILY,
                schedule_time="08:00",
                timerange_days=1,
                output_format="pdf",
                recipients=["admin@company.com"],
                enabled=False  # Disabled by default
            )
            
            # Weekly anomaly analysis
            weekly_report = ScheduledReportConfig(
                report_id="weekly_anomalies",
                report_name="Weekly Anomaly Analysis",
                template_id="anomaly_analysis",
                frequency=ScheduleFrequency.WEEKLY,
                schedule_time="Monday 09:00",
                timerange_days=7,
                output_format="both",
                recipients=["technical@company.com"],
                enabled=False  # Disabled by default
            )
            
            # Monthly model performance report
            monthly_report = ScheduledReportConfig(
                report_id="monthly_models",
                report_name="Monthly Model Performance Report",
                template_id="model_performance",
                frequency=ScheduleFrequency.MONTHLY,
                schedule_time="1st 10:00",
                timerange_days=30,
                output_format="excel",
                recipients=["data-science@company.com"],
                enabled=False  # Disabled by default
            )
            
            self.scheduled_reports["daily_health"] = daily_report
            self.scheduled_reports["weekly_anomalies"] = weekly_report
            self.scheduled_reports["monthly_models"] = monthly_report
            
            self._save_configurations()
            
        except Exception as e:
            self.logger.error(f"Error creating default configurations: {e}")
    
    def _save_configurations(self):
        """Save scheduled report configurations"""
        try:
            config_path = Path("telemetry_monitor/scheduled_reports.json")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'scheduled_reports': [],
                'email_configurations': [],
                'last_updated': datetime.now().isoformat()
            }
            
            # Save scheduled reports
            for config in self.scheduled_reports.values():
                config_dict = asdict(config)
                # Convert enums to strings
                config_dict['frequency'] = config.frequency.value
                
                if config_dict['delivery_config']:
                    delivery_dict = config_dict['delivery_config']
                    delivery_dict['method'] = config.delivery_config.method.value
                    
                    if delivery_dict['email_config']:
                        # Email config is handled separately
                        delivery_dict['email_config'] = None
                
                data['scheduled_reports'].append(config_dict)
            
            # Save email configurations separately for security
            for config_id, email_manager in self.email_managers.items():
                email_dict = asdict(email_manager.config)
                email_dict['config_id'] = config_id
                # Remove sensitive data or encrypt it
                email_dict['password'] = "***ENCRYPTED***"
                data['email_configurations'].append(email_dict)
            
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving scheduled report configurations: {e}")
    
    def add_scheduled_report(self, config: ScheduledReportConfig) -> bool:
        """Add a new scheduled report"""
        try:
            if config.report_id in self.scheduled_reports:
                self.logger.warning(f"Scheduled report {config.report_id} already exists")
                return False
            
            # Calculate next run time
            config.next_run = self._calculate_next_run(config)
            
            self.scheduled_reports[config.report_id] = config
            self._save_configurations()
            
            # Reschedule if scheduler is running
            if self.running:
                self._schedule_report(config)
            
            self.logger.info(f"Added scheduled report: {config.report_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding scheduled report: {e}")
            return False
    
    def update_scheduled_report(self, report_id: str, config: ScheduledReportConfig) -> bool:
        """Update scheduled report configuration"""
        try:
            if report_id not in self.scheduled_reports:
                self.logger.warning(f"Scheduled report {report_id} not found")
                return False
            
            # Calculate next run time
            config.next_run = self._calculate_next_run(config)
            
            self.scheduled_reports[report_id] = config
            self._save_configurations()
            
            # Reschedule if scheduler is running
            if self.running:
                self._clear_schedule()
                self._schedule_all_reports()
            
            self.logger.info(f"Updated scheduled report: {config.report_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating scheduled report: {e}")
            return False
    
    def remove_scheduled_report(self, report_id: str) -> bool:
        """Remove a scheduled report"""
        try:
            if report_id not in self.scheduled_reports:
                self.logger.warning(f"Scheduled report {report_id} not found")
                return False
            
            report_name = self.scheduled_reports[report_id].report_name
            del self.scheduled_reports[report_id]
            self._save_configurations()
            
            # Reschedule if scheduler is running
            if self.running:
                self._clear_schedule()
                self._schedule_all_reports()
            
            self.logger.info(f"Removed scheduled report: {report_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing scheduled report: {e}")
            return False
    
    def start_scheduler(self) -> bool:
        """Start the report scheduler"""
        try:
            if self.running:
                self.logger.warning("Report scheduler already running")
                return True
            
            self.running = True
            
            # Schedule all enabled reports
            self._schedule_all_reports()
            
            # Start scheduler thread
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            
            self.logger.info("Report scheduler started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting report scheduler: {e}")
            return False
    
    def stop_scheduler(self):
        """Stop the report scheduler"""
        try:
            self.logger.info("Stopping report scheduler...")
            self.running = False
            
            # Clear scheduled jobs
            self._clear_schedule()
            
            # Wait for scheduler thread to finish
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)
            
            self.logger.info("Report scheduler stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping report scheduler: {e}")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(5)
    
    def _schedule_all_reports(self):
        """Schedule all enabled reports"""
        for config in self.scheduled_reports.values():
            if config.enabled:
                self._schedule_report(config)
    
    def _schedule_report(self, config: ScheduledReportConfig):
        """Schedule a specific report"""
        try:
            job_func = lambda: self._execute_report(config.report_id)
            
            if config.frequency == ScheduleFrequency.HOURLY:
                # Parse minute from time (e.g., ":30" for 30 minutes past the hour)
                if config.schedule_time.startswith(':'):
                    minute = int(config.schedule_time[1:])
                    schedule.every().hour.at(f":{minute:02d}").do(job_func)
                else:
                    schedule.every().hour.do(job_func)
                    
            elif config.frequency == ScheduleFrequency.DAILY:
                schedule.every().day.at(config.schedule_time).do(job_func)
                
            elif config.frequency == ScheduleFrequency.WEEKLY:
                # Parse day and time (e.g., "Monday 09:00")
                parts = config.schedule_time.split()
                if len(parts) == 2:
                    day, time_str = parts
                    getattr(schedule.every(), day.lower()).at(time_str).do(job_func)
                else:
                    schedule.every().week.do(job_func)
                    
            elif config.frequency == ScheduleFrequency.MONTHLY:
                # For monthly, we'll use a custom check
                schedule.every().day.at("00:00").do(lambda: self._check_monthly_schedule(config.report_id))
            
            self.logger.debug(f"Scheduled report: {config.report_name}")
            
        except Exception as e:
            self.logger.error(f"Error scheduling report {config.report_id}: {e}")
    
    def _clear_schedule(self):
        """Clear all scheduled jobs"""
        schedule.clear()
    
    def _calculate_next_run(self, config: ScheduledReportConfig) -> datetime:
        """Calculate next run time for a report"""
        try:
            now = datetime.now()
            
            if config.frequency == ScheduleFrequency.HOURLY:
                return now + timedelta(hours=1)
                
            elif config.frequency == ScheduleFrequency.DAILY:
                # Parse time
                hour, minute = map(int, config.schedule_time.split(':'))
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                if next_run <= now:
                    next_run += timedelta(days=1)
                    
                return next_run
                
            elif config.frequency == ScheduleFrequency.WEEKLY:
                # This is a simplified calculation
                return now + timedelta(weeks=1)
                
            elif config.frequency == ScheduleFrequency.MONTHLY:
                # Simple monthly calculation
                if now.month == 12:
                    next_month = now.replace(year=now.year + 1, month=1, day=1)
                else:
                    next_month = now.replace(month=now.month + 1, day=1)
                return next_month
            
            return now + timedelta(days=1)  # Default
            
        except Exception as e:
            self.logger.error(f"Error calculating next run time: {e}")
            return datetime.now() + timedelta(hours=1)
    
    def _check_monthly_schedule(self, report_id: str):
        """Check if a monthly report should run"""
        try:
            config = self.scheduled_reports.get(report_id)
            if not config or not config.enabled:
                return
            
            now = datetime.now()
            
            # Parse schedule time (e.g., "1st 10:00", "15th 14:30")
            if "st " in config.schedule_time or "nd " in config.schedule_time or \
               "rd " in config.schedule_time or "th " in config.schedule_time:
                
                day_part, time_part = config.schedule_time.split(' ', 1)
                target_day = int(''.join(filter(str.isdigit, day_part)))
                target_hour, target_minute = map(int, time_part.split(':'))
                
                if (now.day == target_day and 
                    now.hour == target_hour and 
                    now.minute >= target_minute and 
                    now.minute < target_minute + 5):  # 5-minute window
                    
                    self._execute_report(report_id)
                    
        except Exception as e:
            self.logger.error(f"Error checking monthly schedule for {report_id}: {e}")
    
    def _execute_report(self, report_id: str):
        """Execute a scheduled report"""
        try:
            config = self.scheduled_reports.get(report_id)
            if not config or not config.enabled:
                return
            
            # Create execution record
            execution_id = f"{report_id}_{int(datetime.now().timestamp())}"
            execution = ReportExecution(
                execution_id=execution_id,
                report_id=report_id,
                start_time=datetime.now(),
                status=ReportStatus.GENERATING
            )
            
            self.report_executions[execution_id] = execution
            self.logger.info(f"Starting scheduled report execution: {config.report_name}")
            
            try:
                # Calculate time range
                end_time = datetime.now()
                start_time = end_time - timedelta(days=config.timerange_days)
                
                # Create report request
                request = ReportRequest(
                    report_type=config.template_id,
                    template_id=config.template_id,
                    output_format=config.output_format,
                    timerange_start=start_time,
                    timerange_end=end_time,
                    include_sections=config.custom_params.get('include_sections', []),
                    custom_params=config.custom_params
                )
                
                # Generate report
                result = self.report_manager.generate_report(request)
                
                if result.get('success'):
                    execution.status = ReportStatus.COMPLETED
                    execution.output_files = [file_path for _, file_path in result.get('files', [])]
                    
                    # Deliver report
                    if config.delivery_config:
                        delivery_success = self._deliver_report(config, execution)
                        execution.delivery_status = delivery_success
                        
                        if any(delivery_success.values()):
                            execution.status = ReportStatus.SENT
                    
                    # Update config
                    config.last_run = execution.start_time
                    config.next_run = self._calculate_next_run(config)
                    config.run_count += 1
                    
                    self.logger.info(f"Scheduled report completed successfully: {config.report_name}")
                    
                else:
                    execution.status = ReportStatus.FAILED
                    execution.error_message = result.get('error', 'Unknown error')
                    config.failure_count += 1
                    
                    self.logger.error(f"Scheduled report failed: {config.report_name} - {execution.error_message}")
            
            except Exception as e:
                execution.status = ReportStatus.FAILED
                execution.error_message = str(e)
                config.failure_count += 1
                self.logger.error(f"Error executing scheduled report {report_id}: {e}")
            
            finally:
                execution.end_time = datetime.now()
                execution.execution_time_seconds = (execution.end_time - execution.start_time).total_seconds()
                
                # Add to history
                self.execution_history.append(execution)
                
                # Limit history size
                if len(self.execution_history) > 1000:
                    self.execution_history = self.execution_history[-500:]
                
                # Save configurations
                self._save_configurations()
                
        except Exception as e:
            self.logger.error(f"Critical error executing scheduled report {report_id}: {e}")
    
    def _deliver_report(self, config: ScheduledReportConfig, execution: ReportExecution) -> Dict[str, bool]:
        """Deliver report using configured delivery method"""
        delivery_results = {}
        
        try:
            if not config.delivery_config:
                return {}
            
            delivery_config = config.delivery_config
            
            if delivery_config.method == DeliveryMethod.EMAIL:
                delivery_results['email'] = self._deliver_via_email(config, execution)
                
            elif delivery_config.method == DeliveryMethod.FILE_SYSTEM:
                delivery_results['file_system'] = self._deliver_via_file_system(config, execution)
                
            # Add more delivery methods as needed
            
            return delivery_results
            
        except Exception as e:
            self.logger.error(f"Error delivering report: {e}")
            return {'error': False}
    
    def _deliver_via_email(self, config: ScheduledReportConfig, execution: ReportExecution) -> bool:
        """Deliver report via email"""
        try:
            if not config.delivery_config or not config.delivery_config.email_config:
                return False
            
            email_config = config.delivery_config.email_config
            email_manager = EmailManager(email_config)
            
            # Create email content
            subject = f"Scheduled Report: {config.report_name} - {execution.start_time.strftime('%Y-%m-%d')}"
            
            body = f"""
                <html>
                <head></head>
                <body>
                    <h2>{config.report_name}</h2>
                    <p><strong>Generated:</strong> {execution.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Time Range:</strong> {config.timerange_days} days</p>
                    <p><strong>Execution Time:</strong> {execution.execution_time_seconds:.2f} seconds</p>
                    
                    {'<p><strong>Files:</strong></p><ul>' + ''.join([f'<li>{os.path.basename(f)}</li>' for f in execution.output_files]) + '</ul>' if execution.output_files else ''}
                    
                    <p>This is an automated report from the Satellite Telemetry Data Management System.</p>
                    
                    <hr>
                    <small>Generated by STDMS Scheduled Reporting System</small>
                </body>
                </html>
            """
            
            # Send email
            return email_manager.send_report_email(
                recipients=config.recipients,
                subject=subject,
                body=body,
                attachments=execution.output_files
            )
            
        except Exception as e:
            self.logger.error(f"Error delivering report via email: {e}")
            return False
    
    def _deliver_via_file_system(self, config: ScheduledReportConfig, execution: ReportExecution) -> bool:
        """Deliver report to file system location"""
        try:
            if not config.delivery_config or not config.delivery_config.file_path:
                return False
            
            target_dir = Path(config.delivery_config.file_path)
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files to target directory
            import shutil
            for file_path in execution.output_files:
                if os.path.exists(file_path):
                    target_file = target_dir / os.path.basename(file_path)
                    shutil.copy2(file_path, target_file)
            
            self.logger.info(f"Report files copied to: {target_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error delivering report to file system: {e}")
            return False
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status"""
        try:
            status = {
                'running': self.running,
                'total_scheduled_reports': len(self.scheduled_reports),
                'enabled_reports': sum(1 for config in self.scheduled_reports.values() if config.enabled),
                'total_executions': len(self.execution_history),
                'recent_executions': [],
                'scheduled_reports': {},
                'next_runs': []
            }
            
            # Recent executions (last 10)
            recent_executions = sorted(self.execution_history, key=lambda x: x.start_time, reverse=True)[:10]
            for execution in recent_executions:
                status['recent_executions'].append({
                    'execution_id': execution.execution_id,
                    'report_id': execution.report_id,
                    'start_time': execution.start_time.isoformat(),
                    'status': execution.status.value,
                    'execution_time': execution.execution_time_seconds,
                    'output_files': len(execution.output_files),
                    'error_message': execution.error_message
                })
            
            # Scheduled reports status
            for report_id, config in self.scheduled_reports.items():
                status['scheduled_reports'][report_id] = {
                    'name': config.report_name,
                    'template_id': config.template_id,
                    'frequency': config.frequency.value,
                    'schedule_time': config.schedule_time,
                    'enabled': config.enabled,
                    'last_run': config.last_run.isoformat() if config.last_run else None,
                    'next_run': config.next_run.isoformat() if config.next_run else None,
                    'run_count': config.run_count,
                    'failure_count': config.failure_count,
                    'recipients': len(config.recipients)
                }
                
                # Add to next runs list
                if config.enabled and config.next_run:
                    status['next_runs'].append({
                        'report_id': report_id,
                        'report_name': config.report_name,
                        'next_run': config.next_run.isoformat()
                    })
            
            # Sort next runs by time
            status['next_runs'].sort(key=lambda x: x['next_run'])
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting scheduler status: {e}")
            return {'error': str(e)}
    
    def run_report_now(self, report_id: str) -> bool:
        """Run a scheduled report immediately"""
        try:
            config = self.scheduled_reports.get(report_id)
            if not config:
                self.logger.error(f"Scheduled report {report_id} not found")
                return False
            
            # Run in separate thread to avoid blocking
            thread = threading.Thread(target=self._execute_report, args=(report_id,), daemon=True)
            thread.start()
            
            self.logger.info(f"Manual execution started for report: {config.report_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error running report {report_id} now: {e}")
            return False

# Global instance
_report_scheduler = None

def get_report_scheduler() -> ReportScheduler:
    """Get the global report scheduler instance"""
    global _report_scheduler
    if _report_scheduler is None:
        _report_scheduler = ReportScheduler()
    return _report_scheduler

# Convenience functions
def schedule_daily_health_report(recipients: List[str], time: str = "08:00", email_config: EmailConfiguration = None) -> bool:
    """Schedule a daily system health report"""
    scheduler = get_report_scheduler()
    
    delivery_config = None
    if email_config and recipients:
        delivery_config = DeliveryConfiguration(
            method=DeliveryMethod.EMAIL,
            email_config=email_config
        )
    
    config = ScheduledReportConfig(
        report_id=f"daily_health_{int(datetime.now().timestamp())}",
        report_name="Daily System Health Report",
        template_id="system_health",
        frequency=ScheduleFrequency.DAILY,
        schedule_time=time,
        timerange_days=1,
        output_format="pdf",
        delivery_config=delivery_config,
        recipients=recipients,
        enabled=True
    )
    
    return scheduler.add_scheduled_report(config)

def schedule_weekly_anomaly_report(recipients: List[str], day_time: str = "Monday 09:00", email_config: EmailConfiguration = None) -> bool:
    """Schedule a weekly anomaly analysis report"""
    scheduler = get_report_scheduler()
    
    delivery_config = None
    if email_config and recipients:
        delivery_config = DeliveryConfiguration(
            method=DeliveryMethod.EMAIL,
            email_config=email_config
        )
    
    config = ScheduledReportConfig(
        report_id=f"weekly_anomaly_{int(datetime.now().timestamp())}",
        report_name="Weekly Anomaly Analysis Report",
        template_id="anomaly_analysis",
        frequency=ScheduleFrequency.WEEKLY,
        schedule_time=day_time,
        timerange_days=7,
        output_format="both",
        delivery_config=delivery_config,
        recipients=recipients,
        enabled=True
    )
    
    return scheduler.add_scheduled_report(config)