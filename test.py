import os
import time
import keyboard
from pywinauto import Application

PHONE_NUMBER = "9632168509"

def dial_number():
    # Step 1: Open Phone Link with the number prefilled
    os.system(f'start tel:{PHONE_NUMBER}')
    print(f"📱 Dialing {PHONE_NUMBER}...")

    # Step 2: Wait for Phone Link to load
    time.sleep(5)

    try:
        # Step 3: Connect to Phone Link by process
        app = Application(backend="uia").connect(path="PhoneExperienceHost.exe")

        # Step 4: Get the main window
        win = app.top_window()
        win.set_focus()

        # Step 5: Find the Call button via AutomationId
        call_btn = win.child_window(auto_id="ButtonCall", control_type="Button")

        if call_btn.exists():
            call_btn.click_input()
            print("✅ Call button clicked successfully!")
        else:
            print("⚠️ Call button not found.")

    except Exception as e:
        print("❌ Error:", e)

if __name__ == "__main__":
    print("Press 'T' to dial the number...")
    while True:
        if keyboard.is_pressed("t"):
            dial_number()
            break
