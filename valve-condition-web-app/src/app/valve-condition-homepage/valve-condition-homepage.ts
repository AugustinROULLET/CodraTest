import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { APP_CONFIG } from '../config';

@Component({
  selector: 'app-valve-condition-homepage',
  imports: [],
  standalone: true,
  templateUrl: './valve-condition-homepage.html',
  styleUrl: './valve-condition-homepage.scss'
})
export class ValveConditionHomepage {
  private pressureFile: File | null = null;
  private flowFile: File | null = null;
  public outputText: string = "";
  public isTraining: boolean = false;
  constructor(private http: HttpClient) {}


  public onFileSelected(event: Event, type: 'pressure' | 'flow'): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] || null;
    if (file) {
      if (type === 'pressure') {
        this.pressureFile = file;
      } else if (type === 'flow') {
        this.flowFile = file;
      }
    }
  }


  public checkValveCondition(): void {
    // check file content before the prediction request
    if (!this.pressureFile || !this.flowFile) {
      this.outputText = 'Please select both files.';
      return;
    }

    const pressureReader = new FileReader();
    pressureReader.onload = () => {
      const pressureData = this.parseData(pressureReader.result as string);
      if (pressureData.length !== 6000) {
        this.outputText = 'Pressure file must contain 6000 values.';
        return;
      }

      const flowReader = new FileReader();
      flowReader.onload = () => {
        const flowData = this.parseData(flowReader.result as string);
        if (flowData.length !== 600) {
          this.outputText = 'Flow file must contain 600 values.';
          return;
        }
        this.sendToServer({ pressure: pressureData, flow: flowData });
      };
      // load the flowReader reading
      flowReader.readAsText(this.flowFile!);
    };
    // load the pressureReader reading
    pressureReader.readAsText(this.pressureFile!);
  }


  public trainModel() {
    // send a request to train the model
    this.isTraining = true;
    this.outputText = "Training...";
    this.http.post(APP_CONFIG.apiUrl + '/train', {}).subscribe({
      next: _ => {
        this.outputText = "Training done";
        this.isTraining = false;
      },
      error: error => {
        this.outputText = "Error during training: " + (error.message || error);
        this.isTraining = false;
      }
    });
  }


  private parseData(individual: string): number[] {
    return individual
      .split(/\s+/)
      .map(value => parseFloat(value.trim()))
      .filter(value => !isNaN(value));
  }


  private sendToServer(data: { pressure: number[]; flow: number[] }): void {
    // send a prediction request
    this.http.post(APP_CONFIG.apiUrl + '/predict', data).subscribe({
      next: response => this.outputText = "Predicted valve condition: " + response,
      error: error => this.outputText = "Error during prediction: " + error
    });
  }
}
